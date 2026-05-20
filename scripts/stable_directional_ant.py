from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class StableDirectionalRewardConfig:
    # Direction control
    target_yaw: float = 0.0
    w_heading: float = 0.35
    w_lateral_velocity: float = 0.08
    w_lateral_position: float = 0.0
    lateral_position_clip: float = 5.0
    w_yaw_rate: float = 0.03
    target_forward_velocity: float | None = None
    w_forward_velocity: float = 0.0

    # Body stability
    target_height: float | None = None
    w_roll_pitch: float = 0.12
    w_height: float = 0.40
    w_vertical_velocity: float = 0.03

    # Smooth control
    w_action_smooth: float = 0.015


def wrap_angle_rad(x: float) -> float:
    return float((x + np.pi) % (2.0 * np.pi) - np.pi)


def quat_wxyz_to_rpy(q: np.ndarray) -> tuple[float, float, float]:
    """
    Convert MuJoCo free-joint quaternion [w, x, y, z] to roll, pitch, yaw.
    """
    w, x, y, z = [float(v) for v in q]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return float(roll), float(pitch), float(yaw)


class StableDirectionalAntWrapper(gym.Wrapper):
    """
    Reward-shaping wrapper for heading-constrained and body-stable Ant locomotion.

    It keeps the original Ant-v5 reward and adds mild shaping terms:

        r_total = r_ant_base
                - heading error penalty
                - lateral velocity penalty
                - weak lateral position penalty
                - yaw rate penalty
                - optional forward velocity target penalty
                - roll/pitch penalty
                - height tracking penalty
                - vertical velocity penalty
                - action delta penalty

    The goal is not to over-constrain gait, but to reduce side-walking,
    heading drift, body oscillation, and high-frequency action jitter.
    """

    def __init__(
        self,
        env: gym.Env,
        reward_config: StableDirectionalRewardConfig | None = None,
    ):
        super().__init__(env)
        self.cfg = reward_config or StableDirectionalRewardConfig()
        self.prev_action: np.ndarray | None = None
        self.episode_target_height: float | None = None
        self.episode_initial_y: float | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_action = None

        z = self._root_z()
        self.episode_target_height = (
            float(z) if self.cfg.target_height is None else float(self.cfg.target_height)
        )
        self.episode_initial_y = self._root_y()

        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        action_np = np.asarray(action, dtype=np.float64)
        terms = self._compute_reward_terms(action_np)

        shaped_reward = float(base_reward) + float(sum(terms.values()))

        info = dict(info)
        info["base_reward"] = float(base_reward)
        info["stable_shaping_reward"] = float(sum(terms.values()))
        for key, value in terms.items():
            info[f"stable_{key}"] = float(value)

        metrics = self.compute_metrics()
        for key, value in metrics.items():
            info[f"metric_{key}"] = float(value)

        self.prev_action = action_np.copy()

        return obs, shaped_reward, terminated, truncated, info

    def _compute_reward_terms(self, action: np.ndarray) -> dict[str, float]:
        cfg = self.cfg
        metrics = self.compute_metrics()

        yaw_error = wrap_angle_rad(metrics["yaw"] - cfg.target_yaw)

        # Direction stability
        heading_penalty = -cfg.w_heading * (1.0 - math.cos(yaw_error))
        lateral_velocity_penalty = -cfg.w_lateral_velocity * metrics["vy"] ** 2
        initial_y = (
            self.episode_initial_y
            if self.episode_initial_y is not None
            else metrics["y"]
        )
        lateral_position_error = metrics["y"] - initial_y
        lateral_position_error = float(
            np.clip(
                lateral_position_error,
                -cfg.lateral_position_clip,
                cfg.lateral_position_clip,
            )
        )
        lateral_position_penalty = (
            -cfg.w_lateral_position * lateral_position_error**2
        )
        yaw_rate_penalty = -cfg.w_yaw_rate * metrics["yaw_rate"] ** 2
        if cfg.target_forward_velocity is None:
            forward_velocity_penalty = 0.0
        else:
            forward_velocity_error = metrics["vx"] - cfg.target_forward_velocity
            forward_velocity_penalty = -cfg.w_forward_velocity * forward_velocity_error**2

        # Body stability
        roll_pitch_penalty = -cfg.w_roll_pitch * (
            metrics["roll"] ** 2 + metrics["pitch"] ** 2
        )

        target_height = (
            self.episode_target_height
            if self.episode_target_height is not None
            else metrics["z"]
        )
        height_error = metrics["z"] - target_height
        height_penalty = -cfg.w_height * height_error**2

        vertical_velocity_penalty = -cfg.w_vertical_velocity * metrics["vz"] ** 2

        # Smoothness
        if self.prev_action is None:
            action_smooth_penalty = 0.0
        else:
            da = action - self.prev_action
            action_smooth_penalty = -cfg.w_action_smooth * float(np.sum(da * da))

        return {
            "heading": heading_penalty,
            "lateral_velocity": lateral_velocity_penalty,
            "lateral_position": lateral_position_penalty,
            "yaw_rate": yaw_rate_penalty,
            "forward_velocity": forward_velocity_penalty,
            "roll_pitch": roll_pitch_penalty,
            "height": height_penalty,
            "vertical_velocity": vertical_velocity_penalty,
            "action_smooth": action_smooth_penalty,
        }

    def compute_metrics(self) -> dict[str, float]:
        qpos = np.asarray(self.unwrapped.data.qpos, dtype=np.float64)
        qvel = np.asarray(self.unwrapped.data.qvel, dtype=np.float64)

        x = float(qpos[0])
        y = float(qpos[1])
        z = float(qpos[2])
        quat_wxyz = qpos[3:7]
        roll, pitch, yaw = quat_wxyz_to_rpy(quat_wxyz)

        vx = float(qvel[0])
        vy = float(qvel[1])
        vz = float(qvel[2])

        # For MuJoCo free joints, qvel[3:6] are root angular velocity components.
        wx = float(qvel[3])
        wy = float(qvel[4])
        wz = float(qvel[5])

        return {
            "x": x,
            "y": y,
            "z": z,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "roll_rate": wx,
            "pitch_rate": wy,
            "yaw_rate": wz,
            "heading_alignment": math.cos(wrap_angle_rad(yaw - self.cfg.target_yaw)),
        }

    def _root_z(self) -> float:
        return float(self.unwrapped.data.qpos[2])

    def _root_y(self) -> float:
        return float(self.unwrapped.data.qpos[1])


class LateralErrorObservationWrapper(gym.ObservationWrapper):
    """
    Append normalized lateral displacement to the Ant observation.

    Gymnasium Ant-v5 excludes global x/y position from the default observation, so a
    reward term based on y displacement is only partially observable to the policy.
    This wrapper adds clip(y - y_initial) / clip as a compact correction signal.
    """

    def __init__(self, env: gym.Env, clip: float = 5.0):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("LateralErrorObservationWrapper requires a Box observation.")
        if clip <= 0.0:
            raise ValueError("clip must be positive.")

        self.clip = float(clip)
        self.episode_initial_y: float | None = None

        low = np.concatenate(
            [
                np.asarray(env.observation_space.low, dtype=np.float32),
                np.array([-1.0], dtype=np.float32),
            ]
        )
        high = np.concatenate(
            [
                np.asarray(env.observation_space.high, dtype=np.float32),
                np.array([1.0], dtype=np.float32),
            ]
        )
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_initial_y = self._root_y()
        return self.observation(obs), info

    def observation(self, observation):
        y_error = self._root_y() - (
            self.episode_initial_y
            if self.episode_initial_y is not None
            else self._root_y()
        )
        normalized_y_error = float(np.clip(y_error, -self.clip, self.clip) / self.clip)
        return np.concatenate(
            [
                np.asarray(observation, dtype=self.observation_space.dtype),
                np.array([normalized_y_error], dtype=self.observation_space.dtype),
            ]
        )

    def _root_y(self) -> float:
        return float(self.unwrapped.data.qpos[1])


@dataclass(frozen=True)
class ControlledLocomotionRewardConfig:
    # Command tracking
    target_forward_velocity: float = 2.0
    target_lateral_velocity: float = 0.0
    target_yaw_rate: float = 0.0
    target_yaw: float = 0.0
    target_height: float = 0.53
    target_velocity_obs_scale: float = 3.0
    target_yaw_rate_obs_scale: float = 2.0
    randomize_commands: bool = False
    command_forward_velocity_min: float = 1.6
    command_forward_velocity_max: float = 2.2
    command_lateral_velocity_min: float = 0.0
    command_lateral_velocity_max: float = 0.0
    command_yaw_rate_min: float = 0.0
    command_yaw_rate_max: float = 0.0

    # Observation normalization
    lateral_position_clip: float = 5.0
    lateral_position_reward_clip: float = 5.0
    lateral_position_soft_limit: float = 0.0
    include_command_observation: bool = False

    # Positive tracking rewards
    w_alive: float = 0.8
    w_velocity_tracking: float = 1.6
    velocity_tracking_sigma: float = 0.25
    w_lateral_velocity_tracking: float = 0.0
    lateral_velocity_tracking_sigma: float = 0.25
    w_yaw_rate_tracking: float = 0.0
    yaw_rate_tracking_sigma: float = 0.25
    w_heading_tracking: float = 1.2
    heading_tracking_sigma: float = 0.50
    w_course_tracking: float = 0.0
    course_tracking_sigma: float = 0.25
    course_tracking_min_speed: float = 0.25

    # Stability and path penalties
    w_lateral_position: float = 0.035
    w_lateral_velocity: float = 0.10
    w_lateral_away_velocity: float = 0.0
    w_yaw_rate: float = 0.03
    w_roll_pitch: float = 0.12
    w_roll_pitch_rate: float = 0.03
    w_height: float = 0.60
    w_vertical_velocity: float = 0.04

    # Control quality penalties
    w_action_energy: float = 0.02
    w_action_rate: float = 0.05
    w_action_accel: float = 0.015


class ControlledLocomotionAntWrapper(gym.Wrapper):
    """
    Command-tracking locomotion objective for v3.

    Unlike StableDirectionalAntWrapper, this wrapper does not add penalties to
    the original Ant forward reward. It replaces the reward with a robotics-style
    command tracking objective: target forward/lateral velocity, heading/yaw-rate
    alignment, velocity-vector alignment, lateral centering, body stability, and
    smooth control.

    The observation is also augmented with compact command/error signals:
    normalized lateral displacement, sin/cos heading error, normalized target
    forward velocity, and the previous action.
    """

    def __init__(
        self,
        env: gym.Env,
        reward_config: ControlledLocomotionRewardConfig | None = None,
    ):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("ControlledLocomotionAntWrapper requires a Box observation.")

        self.cfg = reward_config or ControlledLocomotionRewardConfig()
        self.episode_initial_y: float | None = None
        self.current_target_forward_velocity = self.cfg.target_forward_velocity
        self.current_target_lateral_velocity = self.cfg.target_lateral_velocity
        self.current_target_yaw_rate = self.cfg.target_yaw_rate
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float64)
        self.prev_prev_action = np.zeros(self.action_space.shape, dtype=np.float64)

        compact_signal_dim = 4
        if self.cfg.include_command_observation:
            compact_signal_dim += 2

        extra_low = np.concatenate(
            [
                -np.ones(compact_signal_dim, dtype=np.float32),
                np.asarray(self.action_space.low, dtype=np.float32),
            ]
        )
        extra_high = np.concatenate(
            [
                np.ones(compact_signal_dim, dtype=np.float32),
                np.asarray(self.action_space.high, dtype=np.float32),
            ]
        )
        low = np.concatenate(
            [
                np.asarray(env.observation_space.low, dtype=np.float32),
                extra_low,
            ]
        )
        high = np.concatenate(
            [
                np.asarray(env.observation_space.high, dtype=np.float32),
                extra_high,
            ]
        )
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_initial_y = self._root_y()
        self._sample_command()
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float64)
        self.prev_prev_action = np.zeros(self.action_space.shape, dtype=np.float64)
        return self._augment_observation(obs), info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        action_np = np.asarray(action, dtype=np.float64)
        metrics = self.compute_metrics()
        terms = self._compute_reward_terms(
            action=action_np,
            metrics=metrics,
            terminated=terminated,
        )
        controlled_reward = float(sum(terms.values()))

        info = dict(info)
        info["base_reward"] = float(base_reward)
        info["controlled_reward"] = controlled_reward
        for key, value in terms.items():
            info[f"controlled_{key}"] = float(value)
        for key, value in metrics.items():
            info[f"metric_{key}"] = float(value)

        self.prev_prev_action = self.prev_action.copy()
        self.prev_action = action_np.copy()

        return self._augment_observation(obs), controlled_reward, terminated, truncated, info

    def _augment_observation(self, observation):
        metrics = self.compute_metrics()
        normalized_y_error = self._normalized_lateral_position_error(metrics["y"])
        yaw_error = wrap_angle_rad(metrics["yaw"] - self.cfg.target_yaw)
        target_vx = self._normalized_target_velocity(
            self.current_target_forward_velocity
        )
        compact_signals = [
            normalized_y_error,
            math.sin(yaw_error),
            math.cos(yaw_error),
            target_vx,
        ]
        if self.cfg.include_command_observation:
            compact_signals.extend(
                [
                    self._normalized_target_velocity(
                        self.current_target_lateral_velocity
                    ),
                    self._normalized_target_yaw_rate(),
                ]
            )
        extra = np.concatenate(
            [
                np.array(compact_signals, dtype=self.observation_space.dtype),
                self.prev_action.astype(self.observation_space.dtype),
            ]
        )
        return np.concatenate(
            [
                np.asarray(observation, dtype=self.observation_space.dtype),
                extra,
            ]
        )

    def _compute_reward_terms(
        self,
        action: np.ndarray,
        metrics: dict[str, float],
        terminated: bool,
    ) -> dict[str, float]:
        cfg = self.cfg

        vx_error = metrics["vx"] - self.current_target_forward_velocity
        velocity_tracking = cfg.w_velocity_tracking * math.exp(
            -(vx_error**2) / max(cfg.velocity_tracking_sigma, 1e-9)
        )
        vy_error = metrics["vy"] - self.current_target_lateral_velocity
        lateral_velocity_tracking = cfg.w_lateral_velocity_tracking * math.exp(
            -(vy_error**2) / max(cfg.lateral_velocity_tracking_sigma, 1e-9)
        )
        yaw_rate_error = metrics["yaw_rate"] - self.current_target_yaw_rate
        yaw_rate_tracking = cfg.w_yaw_rate_tracking * math.exp(
            -(yaw_rate_error**2) / max(cfg.yaw_rate_tracking_sigma, 1e-9)
        )

        yaw_error = wrap_angle_rad(metrics["yaw"] - cfg.target_yaw)
        heading_tracking = cfg.w_heading_tracking * math.exp(
            -(yaw_error**2) / max(cfg.heading_tracking_sigma, 1e-9)
        )

        course_tracking = 0.0
        if metrics["speed_xy"] >= cfg.course_tracking_min_speed:
            course_tracking = cfg.w_course_tracking * math.exp(
                -(metrics["course_error"] ** 2) / max(cfg.course_tracking_sigma, 1e-9)
            )

        y_error = self._raw_lateral_position_error(metrics["y"])
        lateral_position_penalty = -cfg.w_lateral_position * (
            self._lateral_position_loss(y_error)
        )
        lateral_away_velocity_penalty = -cfg.w_lateral_away_velocity * max(
            0.0,
            y_error * metrics["vy"],
        )

        roll_pitch_penalty = -cfg.w_roll_pitch * (
            metrics["roll"] ** 2 + metrics["pitch"] ** 2
        )
        roll_pitch_rate_penalty = -cfg.w_roll_pitch_rate * (
            metrics["roll_rate"] ** 2 + metrics["pitch_rate"] ** 2
        )
        height_error = metrics["z"] - cfg.target_height
        height_penalty = -cfg.w_height * height_error**2

        action_rate = action - self.prev_action
        action_accel = action - 2.0 * self.prev_action + self.prev_prev_action

        return {
            "alive": cfg.w_alive if not terminated else 0.0,
            "velocity_tracking": velocity_tracking,
            "lateral_velocity_tracking": lateral_velocity_tracking,
            "yaw_rate_tracking": yaw_rate_tracking,
            "heading_tracking": heading_tracking,
            "course_tracking": course_tracking,
            "lateral_position": lateral_position_penalty,
            "lateral_velocity": -cfg.w_lateral_velocity * vy_error**2,
            "lateral_away_velocity": lateral_away_velocity_penalty,
            "yaw_rate": -cfg.w_yaw_rate * yaw_rate_error**2,
            "roll_pitch": roll_pitch_penalty,
            "roll_pitch_rate": roll_pitch_rate_penalty,
            "height": height_penalty,
            "vertical_velocity": -cfg.w_vertical_velocity * metrics["vz"] ** 2,
            "action_energy": -cfg.w_action_energy * float(np.sum(action * action)),
            "action_rate": -cfg.w_action_rate * float(np.sum(action_rate * action_rate)),
            "action_accel": -cfg.w_action_accel * float(np.sum(action_accel * action_accel)),
        }

    def compute_metrics(self) -> dict[str, float]:
        qpos = np.asarray(self.unwrapped.data.qpos, dtype=np.float64)
        qvel = np.asarray(self.unwrapped.data.qvel, dtype=np.float64)

        roll, pitch, yaw = quat_wxyz_to_rpy(qpos[3:7])
        vx = float(qvel[0])
        vy = float(qvel[1])
        speed_xy = float(math.hypot(vx, vy))
        target_course_yaw = self._target_course_yaw()
        course_yaw = target_course_yaw
        if speed_xy > 1e-9:
            course_yaw = math.atan2(vy, vx)
        course_error = wrap_angle_rad(course_yaw - target_course_yaw)

        return {
            "x": float(qpos[0]),
            "y": float(qpos[1]),
            "z": float(qpos[2]),
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "vx": vx,
            "vy": vy,
            "vz": float(qvel[2]),
            "roll_rate": float(qvel[3]),
            "pitch_rate": float(qvel[4]),
            "yaw_rate": float(qvel[5]),
            "heading_alignment": math.cos(wrap_angle_rad(yaw - self.cfg.target_yaw)),
            "speed_xy": speed_xy,
            "course_yaw": float(course_yaw),
            "course_error": float(course_error),
            "course_alignment": math.cos(course_error),
            "target_course_yaw": float(target_course_yaw),
            "target_vx": float(self.current_target_forward_velocity),
            "target_vy": float(self.current_target_lateral_velocity),
            "target_yaw_rate": float(self.current_target_yaw_rate),
        }

    def _raw_lateral_position_error(self, y: float) -> float:
        initial_y = self.episode_initial_y if self.episode_initial_y is not None else y
        return float(y - initial_y)

    def _normalized_lateral_position_error(self, y: float) -> float:
        clip = max(self.cfg.lateral_position_clip, 1e-9)
        return float(np.clip(self._raw_lateral_position_error(y), -clip, clip) / clip)

    def _lateral_position_loss(self, y_error: float) -> float:
        reward_clip = self.cfg.lateral_position_reward_clip
        if reward_clip > 0.0:
            clipped_y_error = float(np.clip(y_error, -reward_clip, reward_clip))
            return clipped_y_error**2

        soft_limit = self.cfg.lateral_position_soft_limit
        if soft_limit <= 0.0:
            return y_error**2

        abs_y_error = abs(y_error)
        if abs_y_error <= soft_limit:
            return y_error**2
        return float(
            soft_limit * (2.0 * abs_y_error - soft_limit)
        )

    def _normalized_target_velocity(self, velocity: float) -> float:
        scale = max(self.cfg.target_velocity_obs_scale, 1e-9)
        return float(np.clip(velocity / scale, -1.0, 1.0))

    def _normalized_target_yaw_rate(self) -> float:
        scale = max(self.cfg.target_yaw_rate_obs_scale, 1e-9)
        return float(np.clip(self.current_target_yaw_rate / scale, -1.0, 1.0))

    def _target_course_yaw(self) -> float:
        command_speed = math.hypot(
            self.current_target_forward_velocity,
            self.current_target_lateral_velocity,
        )
        if command_speed <= 1e-9:
            return self.cfg.target_yaw
        return float(
            math.atan2(
                self.current_target_lateral_velocity,
                self.current_target_forward_velocity,
            )
        )

    def _sample_command(self) -> None:
        if not self.cfg.randomize_commands:
            self.current_target_forward_velocity = self.cfg.target_forward_velocity
            self.current_target_lateral_velocity = self.cfg.target_lateral_velocity
            self.current_target_yaw_rate = self.cfg.target_yaw_rate
            return

        rng = self.unwrapped.np_random
        self.current_target_forward_velocity = float(
            rng.uniform(
                self.cfg.command_forward_velocity_min,
                self.cfg.command_forward_velocity_max,
            )
        )
        self.current_target_lateral_velocity = float(
            rng.uniform(
                self.cfg.command_lateral_velocity_min,
                self.cfg.command_lateral_velocity_max,
            )
        )
        self.current_target_yaw_rate = float(
            rng.uniform(
                self.cfg.command_yaw_rate_min,
                self.cfg.command_yaw_rate_max,
            )
        )

    def _root_y(self) -> float:
        return float(self.unwrapped.data.qpos[1])


@dataclass(frozen=True)
class WellTrainedLocomotionRewardConfig:
    """
    Body-frame command tracking objective (v3d) following legged_gym conventions.

    Commands are expressed in the robot's body frame. With yaw kept near zero by
    orientation/heading penalties, body-frame and world-frame velocities coincide,
    but the body-frame formulation breaks the crab-walk / yawed-walk degeneracy
    that earlier reward formulations admitted as separate local optima.
    """

    # Command (body frame)
    target_forward_velocity: float = 2.0
    target_lateral_velocity: float = 0.0
    target_yaw_rate: float = 0.0
    target_height: float = 0.53

    # Command randomization (off by default; reserved for future curriculum work)
    randomize_commands: bool = False
    command_forward_velocity_min: float = 1.6
    command_forward_velocity_max: float = 2.2
    command_lateral_velocity_min: float = 0.0
    command_lateral_velocity_max: float = 0.0
    command_yaw_rate_min: float = 0.0
    command_yaw_rate_max: float = 0.0

    # Observation normalization
    velocity_obs_scale: float = 3.0
    include_command_observation: bool = False

    # Positive tracking rewards (body frame)
    w_alive: float = 0.10
    w_track_vx: float = 1.50
    w_track_vy: float = 1.50
    w_track_omega_z: float = 0.50
    sigma_track_v: float = 0.50
    sigma_track_omega: float = 0.50

    # Linear progress reward in body frame. Breaks the "stand still" local
    # optimum at the start of training when w_track_vx alone has near-zero
    # gradient far from the velocity target (legged_gym-style training escapes
    # this via massive env parallelism; with O(8) envs we instead supply a
    # linear progress signal that saturates at the target).
    w_progress_vx: float = 2.0

    # Absolute world-frame heading alignment. Pure body-frame command tracking
    # is yaw-rotation-invariant: the policy can yaw the body and still satisfy
    # vx_body/vy_body. This reward anchors the body to the world +x axis (or
    # any commanded target_yaw) so "go forward" means "go in +x world frame".
    w_heading_alignment: float = 1.0
    sigma_heading_alignment: float = 0.50
    target_yaw: float = 0.0

    # Posture penalties (legged_gym-style). Orientation/base-height kept light
    # so the policy is not penalized into a "stand still" attractor; tipover
    # is suppressed jointly by lin_vel_z, ang_vel_xy, and the Ant terminator.
    w_lin_vel_z: float = 2.0
    w_ang_vel_xy: float = 0.05
    w_orientation: float = 1.0
    w_base_height: float = 1.0

    # Accumulated lateral position drift penalty. Independent of body-frame
    # vy tracking: stops the slow asymmetric drift that body-frame velocity
    # tracking alone cannot suppress (it averages out gait-cycle wiggle while
    # leaving a small mean bias intact).
    w_lateral_position: float = 0.05
    lateral_position_clip: float = 5.0

    # Control quality penalties
    w_action_rate: float = 0.05
    w_action_accel: float = 0.02
    w_action_energy: float = 0.005
    w_dof_vel: float = 0.001


class WellTrainedLocomotionAntWrapper(gym.Wrapper):
    """
    Body-frame command-tracking objective for v3d (well-trained locomotion).

    Differences from ControlledLocomotionAntWrapper (v3a/b/c):

      - Tracks vx_body, vy_body, omega_z against constant body-frame commands
        using exp(-error^2 / sigma^2) reward shape (legged_gym convention).
      - Penalizes orientation (roll, pitch), base height deviation, vertical
        linear velocity, and xy angular velocity directly.
      - Includes joint velocity, action energy, action rate, and action
        acceleration smoothness penalties.
      - No world-frame lateral position penalty or course-tracking heuristics:
        crab-walk and yawed-walk are both naturally suppressed by penalizing
        body-frame vy and roll/pitch orientation.

    The observation is augmented with body-frame linear velocity (vx_body,
    vy_body), heading sin/cos, and the previous action so the policy sees the
    signals it must minimize.
    """

    def __init__(
        self,
        env: gym.Env,
        reward_config: WellTrainedLocomotionRewardConfig | None = None,
    ):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError(
                "WellTrainedLocomotionAntWrapper requires a Box observation."
            )

        self.cfg = reward_config or WellTrainedLocomotionRewardConfig()
        self.current_target_vx = self.cfg.target_forward_velocity
        self.current_target_vy = self.cfg.target_lateral_velocity
        self.current_target_omega_z = self.cfg.target_yaw_rate
        self.episode_initial_y: float | None = None
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float64)
        self.prev_prev_action = np.zeros(self.action_space.shape, dtype=np.float64)

        # 4 compact signals + 8 prev action = 12 extra dims (no commands by default)
        compact_signal_dim = 4
        if self.cfg.include_command_observation:
            compact_signal_dim += 3

        extra_low = np.concatenate(
            [
                -np.ones(compact_signal_dim, dtype=np.float32),
                np.asarray(self.action_space.low, dtype=np.float32),
            ]
        )
        extra_high = np.concatenate(
            [
                np.ones(compact_signal_dim, dtype=np.float32),
                np.asarray(self.action_space.high, dtype=np.float32),
            ]
        )
        low = np.concatenate(
            [
                np.asarray(env.observation_space.low, dtype=np.float32),
                extra_low,
            ]
        )
        high = np.concatenate(
            [
                np.asarray(env.observation_space.high, dtype=np.float32),
                extra_high,
            ]
        )
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._sample_command()
        self.episode_initial_y = float(self.unwrapped.data.qpos[1])
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float64)
        self.prev_prev_action = np.zeros(self.action_space.shape, dtype=np.float64)
        return self._augment_observation(obs), info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        action_np = np.asarray(action, dtype=np.float64)
        metrics = self.compute_metrics()
        terms = self._compute_reward_terms(
            action=action_np,
            metrics=metrics,
            terminated=terminated,
        )
        reward = float(sum(terms.values()))

        info = dict(info)
        info["base_reward"] = float(base_reward)
        info["well_trained_reward"] = reward
        for key, value in terms.items():
            info[f"well_trained_{key}"] = float(value)
        for key, value in metrics.items():
            info[f"metric_{key}"] = float(value)

        self.prev_prev_action = self.prev_action.copy()
        self.prev_action = action_np.copy()

        return self._augment_observation(obs), reward, terminated, truncated, info

    def _augment_observation(self, observation):
        metrics = self.compute_metrics()
        vx_body_n = self._normalize_velocity(metrics["vx_body"])
        vy_body_n = self._normalize_velocity(metrics["vy_body"])
        yaw = metrics["yaw"]
        compact_signals = [
            vx_body_n,
            vy_body_n,
            math.sin(yaw),
            math.cos(yaw),
        ]
        if self.cfg.include_command_observation:
            compact_signals.extend(
                [
                    self._normalize_velocity(self.current_target_vx),
                    self._normalize_velocity(self.current_target_vy),
                    float(np.clip(self.current_target_omega_z / 2.0, -1.0, 1.0)),
                ]
            )
        extra = np.concatenate(
            [
                np.array(compact_signals, dtype=self.observation_space.dtype),
                self.prev_action.astype(self.observation_space.dtype),
            ]
        )
        return np.concatenate(
            [
                np.asarray(observation, dtype=self.observation_space.dtype),
                extra,
            ]
        )

    def _compute_reward_terms(
        self,
        action: np.ndarray,
        metrics: dict[str, float],
        terminated: bool,
    ) -> dict[str, float]:
        cfg = self.cfg

        vx_err = metrics["vx_body"] - self.current_target_vx
        vy_err = metrics["vy_body"] - self.current_target_vy
        omega_z_err = metrics["yaw_rate"] - self.current_target_omega_z

        sigma_v2 = max(cfg.sigma_track_v, 1e-9) ** 2
        sigma_o2 = max(cfg.sigma_track_omega, 1e-9) ** 2

        track_vx = cfg.w_track_vx * math.exp(-(vx_err**2) / sigma_v2)
        track_vy = cfg.w_track_vy * math.exp(-(vy_err**2) / sigma_v2)
        track_omega_z = cfg.w_track_omega_z * math.exp(-(omega_z_err**2) / sigma_o2)

        target_vx = self.current_target_vx
        if target_vx > 0.0:
            progress_vx = cfg.w_progress_vx * max(
                0.0, min(metrics["vx_body"], target_vx)
            )
        elif target_vx < 0.0:
            progress_vx = cfg.w_progress_vx * min(
                0.0, max(metrics["vx_body"], target_vx)
            )
        else:
            progress_vx = 0.0

        yaw_err = wrap_angle_rad(metrics["yaw"] - cfg.target_yaw)
        sigma_h2 = max(cfg.sigma_heading_alignment, 1e-9) ** 2
        heading_alignment_reward = cfg.w_heading_alignment * math.exp(
            -(yaw_err**2) / sigma_h2
        )

        initial_y = (
            self.episode_initial_y
            if self.episode_initial_y is not None
            else metrics["y"]
        )
        y_error = metrics["y"] - initial_y
        clipped_y_error = float(
            np.clip(y_error, -cfg.lateral_position_clip, cfg.lateral_position_clip)
        )
        lateral_position_pen = -cfg.w_lateral_position * clipped_y_error**2

        lin_vel_z_pen = -cfg.w_lin_vel_z * metrics["vz"] ** 2
        ang_vel_xy_pen = -cfg.w_ang_vel_xy * (
            metrics["roll_rate"] ** 2 + metrics["pitch_rate"] ** 2
        )
        orientation_pen = -cfg.w_orientation * (
            metrics["roll"] ** 2 + metrics["pitch"] ** 2
        )
        height_err = metrics["z"] - cfg.target_height
        base_height_pen = -cfg.w_base_height * height_err**2

        action_rate = action - self.prev_action
        action_accel = action - 2.0 * self.prev_action + self.prev_prev_action
        action_rate_pen = -cfg.w_action_rate * float(np.sum(action_rate * action_rate))
        action_accel_pen = -cfg.w_action_accel * float(
            np.sum(action_accel * action_accel)
        )
        action_energy_pen = -cfg.w_action_energy * float(np.sum(action * action))

        qvel = np.asarray(self.unwrapped.data.qvel, dtype=np.float64)
        dof_vel = qvel[6:]
        dof_vel_pen = -cfg.w_dof_vel * float(np.sum(dof_vel * dof_vel))

        return {
            "alive": cfg.w_alive if not terminated else 0.0,
            "track_vx": track_vx,
            "track_vy": track_vy,
            "track_omega_z": track_omega_z,
            "progress_vx": progress_vx,
            "heading_alignment": heading_alignment_reward,
            "lateral_position": lateral_position_pen,
            "lin_vel_z": lin_vel_z_pen,
            "ang_vel_xy": ang_vel_xy_pen,
            "orientation": orientation_pen,
            "base_height": base_height_pen,
            "action_rate": action_rate_pen,
            "action_accel": action_accel_pen,
            "action_energy": action_energy_pen,
            "dof_vel": dof_vel_pen,
        }

    def compute_metrics(self) -> dict[str, float]:
        qpos = np.asarray(self.unwrapped.data.qpos, dtype=np.float64)
        qvel = np.asarray(self.unwrapped.data.qvel, dtype=np.float64)

        roll, pitch, yaw = quat_wxyz_to_rpy(qpos[3:7])
        vx_world = float(qvel[0])
        vy_world = float(qvel[1])

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        vx_body = cos_yaw * vx_world + sin_yaw * vy_world
        vy_body = -sin_yaw * vx_world + cos_yaw * vy_world

        return {
            "x": float(qpos[0]),
            "y": float(qpos[1]),
            "z": float(qpos[2]),
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "vx": vx_world,
            "vy": vy_world,
            "vz": float(qvel[2]),
            "vx_body": float(vx_body),
            "vy_body": float(vy_body),
            "roll_rate": float(qvel[3]),
            "pitch_rate": float(qvel[4]),
            "yaw_rate": float(qvel[5]),
            "heading_alignment": cos_yaw,
            "target_vx": float(self.current_target_vx),
            "target_vy": float(self.current_target_vy),
            "target_omega_z": float(self.current_target_omega_z),
        }

    def _normalize_velocity(self, v: float) -> float:
        scale = max(self.cfg.velocity_obs_scale, 1e-9)
        return float(np.clip(v / scale, -1.0, 1.0))

    def _sample_command(self) -> None:
        if not self.cfg.randomize_commands:
            self.current_target_vx = self.cfg.target_forward_velocity
            self.current_target_vy = self.cfg.target_lateral_velocity
            self.current_target_omega_z = self.cfg.target_yaw_rate
            return

        rng = self.unwrapped.np_random
        self.current_target_vx = float(
            rng.uniform(
                self.cfg.command_forward_velocity_min,
                self.cfg.command_forward_velocity_max,
            )
        )
        self.current_target_vy = float(
            rng.uniform(
                self.cfg.command_lateral_velocity_min,
                self.cfg.command_lateral_velocity_max,
            )
        )
        self.current_target_omega_z = float(
            rng.uniform(
                self.cfg.command_yaw_rate_min,
                self.cfg.command_yaw_rate_max,
            )
        )


@dataclass(frozen=True)
class PushDisturbanceConfig:
    """
    External push disturbance applied to the torso during training.

    The wrapper applies a force F (N) to the torso body in a random xy
    direction for `push_duration_steps` MuJoCo sub-steps, then idles for
    a sampled inter-push interval before the next event. The maximum force
    magnitude ramps linearly with per-env step count (curriculum), which
    avoids destabilising the warm-started locomotion policy at the start.
    """

    enabled: bool = True
    torso_body_name: str = "torso"
    push_force_max: float = 10.0      # N, at end of curriculum
    push_duration_steps: int = 5       # MuJoCo sub-steps push is held (held constant)
    push_interval_steps_min: int = 500   # min steps between pushes (5s @ 100Hz)
    push_interval_steps_max: int = 1000  # max steps between pushes (10s @ 100Hz)
    curriculum_ramp_steps: int = 300_000  # per-env steps to reach push_force_max
    initial_quiet_steps: int = 10_000     # warm-up: no pushes for first N env steps

    # Phase 2b stress disturbances (default off; set max > 0 to enable).
    # All three (force, torque, duration) share the same curriculum progress so
    # the difficulty ramps coherently.
    push_torque_z_max: float = 0.0     # N·m, yaw twist around torso z-axis
    push_duration_max_steps: int = 0   # if > 0, duration is ramped from push_duration_steps to this value


class PushDisturbanceWrapper(gym.Wrapper):
    """
    Apply random xy push forces to the Ant torso during step().

    Schedule per env:
      - Starts in `initial_quiet_steps` quiet phase (no pushes).
      - Then every push_interval_steps in [min,max] uniform, a new push is
        scheduled with magnitude sampled from [0, current_max] (random xy dir).
      - The push is applied for push_duration_steps consecutive sim steps via
        data.xfrc_applied[torso_id, :3].
      - current_max ramps linearly from 0 to push_force_max across
        curriculum_ramp_steps (counted in env steps, persistent across resets).

    Observation: unchanged. The policy must react via proprioception.
    Reward: unchanged.
    Info: adds push_force_magnitude, push_force_dir_x, push_force_dir_y
    so eval scripts can attribute disturbances.
    """

    def __init__(
        self,
        env: gym.Env,
        config: PushDisturbanceConfig | None = None,
    ):
        super().__init__(env)
        self.cfg = config or PushDisturbanceConfig()

        model = self.unwrapped.model
        torso_id = None
        for i in range(model.nbody):
            if model.body(i).name == self.cfg.torso_body_name:
                torso_id = i
                break
        if torso_id is None:
            raise ValueError(
                f"Body '{self.cfg.torso_body_name}' not found in MuJoCo model."
            )
        self._torso_id = torso_id

        # Persistent step counter across resets (per-env, since each env
        # has its own wrapper instance under VecEnv).
        self._env_step_count = 0

        # Scheduling state (reset each episode)
        self._next_push_step = 0
        self._push_remaining_steps = 0
        self._current_force_xy = np.zeros(2, dtype=np.float64)
        self._current_force_magnitude = 0.0
        self._current_torque_z = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Clear any residual force at episode start
        self.unwrapped.data.xfrc_applied[self._torso_id, :] = 0.0
        # Schedule the first push relative to the start of this episode
        self._next_push_step = self._env_step_count + self._sample_interval()
        self._push_remaining_steps = 0
        self._current_force_xy[:] = 0.0
        self._current_force_magnitude = 0.0
        self._current_torque_z = 0.0
        return obs, info

    def step(self, action):
        if self.cfg.enabled:
            self._maybe_start_push()
            self._apply_active_force()
        else:
            self.unwrapped.data.xfrc_applied[self._torso_id, :] = 0.0

        obs, reward, terminated, truncated, info = self.env.step(action)

        info = dict(info)
        info["push_force_magnitude"] = float(self._current_force_magnitude)
        info["push_force_dir_x"] = float(self._current_force_xy[0] / max(self._current_force_magnitude, 1e-9))
        info["push_force_dir_y"] = float(self._current_force_xy[1] / max(self._current_force_magnitude, 1e-9))
        info["push_torque_z"] = float(self._current_torque_z)
        info["push_curriculum_force_max"] = float(self._current_max_force())

        self._env_step_count += 1
        if self._push_remaining_steps > 0:
            self._push_remaining_steps -= 1
            if self._push_remaining_steps == 0:
                self.unwrapped.data.xfrc_applied[self._torso_id, :] = 0.0
                self._current_force_magnitude = 0.0
                self._current_force_xy[:] = 0.0
                self._current_torque_z = 0.0
                self._next_push_step = self._env_step_count + self._sample_interval()

        return obs, reward, terminated, truncated, info

    def _curriculum_progress(self) -> float:
        cfg = self.cfg
        if self._env_step_count < cfg.initial_quiet_steps:
            return 0.0
        progress = (self._env_step_count - cfg.initial_quiet_steps) / max(
            cfg.curriculum_ramp_steps, 1
        )
        return float(np.clip(progress, 0.0, 1.0))

    def _current_max_force(self) -> float:
        return self.cfg.push_force_max * self._curriculum_progress()

    def _current_max_torque_z(self) -> float:
        return self.cfg.push_torque_z_max * self._curriculum_progress()

    def _current_push_duration(self) -> int:
        cfg = self.cfg
        if cfg.push_duration_max_steps <= cfg.push_duration_steps:
            return cfg.push_duration_steps
        progress = self._curriculum_progress()
        duration = cfg.push_duration_steps + progress * (
            cfg.push_duration_max_steps - cfg.push_duration_steps
        )
        return int(round(duration))

    def _sample_interval(self) -> int:
        rng = self.unwrapped.np_random
        return int(
            rng.integers(
                self.cfg.push_interval_steps_min,
                self.cfg.push_interval_steps_max + 1,
            )
        )

    def _maybe_start_push(self) -> None:
        if self._push_remaining_steps > 0:
            return
        if self._env_step_count < self._next_push_step:
            return

        max_f = self._current_max_force()
        max_tau = self._current_max_torque_z()
        if max_f <= 0.0 and max_tau <= 0.0:
            self._next_push_step = self._env_step_count + self._sample_interval()
            return

        rng = self.unwrapped.np_random
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        magnitude = float(rng.uniform(0.0, max_f)) if max_f > 0.0 else 0.0
        self._current_force_xy[0] = magnitude * math.cos(theta)
        self._current_force_xy[1] = magnitude * math.sin(theta)
        self._current_force_magnitude = magnitude
        if max_tau > 0.0:
            self._current_torque_z = float(rng.uniform(-max_tau, max_tau))
        else:
            self._current_torque_z = 0.0
        self._push_remaining_steps = self._current_push_duration()

    def _apply_active_force(self) -> None:
        xfrc = self.unwrapped.data.xfrc_applied
        if self._push_remaining_steps > 0:
            xfrc[self._torso_id, 0] = self._current_force_xy[0]
            xfrc[self._torso_id, 1] = self._current_force_xy[1]
            xfrc[self._torso_id, 2] = 0.0
            xfrc[self._torso_id, 3] = 0.0
            xfrc[self._torso_id, 4] = 0.0
            xfrc[self._torso_id, 5] = self._current_torque_z
        else:
            xfrc[self._torso_id, :] = 0.0


@dataclass(frozen=True)
class DomainRandomizationConfig:
    """
    Per-episode randomization of dynamics + per-step action noise.

    Scales are sampled at every reset and applied multiplicatively to the
    original MuJoCo model parameters (which are restored from a snapshot
    captured at wrapper construction). Action noise is additive Gaussian
    in the [-1, 1] normalized action space.
    """

    enabled: bool = True
    torso_body_name: str = "torso"

    mass_scale_min: float = 0.8
    mass_scale_max: float = 1.2
    friction_scale_min: float = 0.5
    friction_scale_max: float = 1.5
    damping_scale_min: float = 0.8
    damping_scale_max: float = 1.2
    motor_scale_min: float = 0.85
    motor_scale_max: float = 1.15
    action_noise_std: float = 0.02


class DomainRandomizationWrapper(gym.Wrapper):
    """
    Randomize torso mass, geom friction, joint damping, motor gain, and inject
    action noise. Originals are snapshotted at construction and restored before
    each new scaling is applied.
    """

    def __init__(
        self,
        env: gym.Env,
        config: DomainRandomizationConfig | None = None,
    ):
        super().__init__(env)
        self.cfg = config or DomainRandomizationConfig()

        model = self.unwrapped.model

        torso_id = None
        for i in range(model.nbody):
            if model.body(i).name == self.cfg.torso_body_name:
                torso_id = i
                break
        if torso_id is None:
            raise ValueError(
                f"Body '{self.cfg.torso_body_name}' not found in MuJoCo model."
            )
        self._torso_id = torso_id

        self._original_torso_mass = float(model.body_mass[torso_id])
        self._original_geom_friction = np.array(model.geom_friction, copy=True)
        self._original_dof_damping = np.array(model.dof_damping, copy=True)
        # actuator_gainprm has shape (n_actuators, n_params); the first column
        # is the gain (proportional to motor strength) for affine actuators.
        self._original_actuator_gainprm = np.array(model.actuator_gainprm, copy=True)

        self._current_motor_scale = 1.0
        self._current_mass_scale = 1.0
        self._current_friction_scale = 1.0
        self._current_damping_scale = 1.0

    def reset(self, **kwargs):
        if self.cfg.enabled:
            self._sample_and_apply_dynamics()
        else:
            self._restore_originals()
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.cfg.enabled and self.cfg.action_noise_std > 0.0:
            rng = self.unwrapped.np_random
            noise = rng.normal(0.0, self.cfg.action_noise_std, size=action.shape)
            action = np.clip(
                np.asarray(action, dtype=np.float64) + noise,
                self.action_space.low,
                self.action_space.high,
            )
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["dr_mass_scale"] = float(self._current_mass_scale)
        info["dr_friction_scale"] = float(self._current_friction_scale)
        info["dr_damping_scale"] = float(self._current_damping_scale)
        info["dr_motor_scale"] = float(self._current_motor_scale)
        return obs, reward, terminated, truncated, info

    def _sample_and_apply_dynamics(self) -> None:
        rng = self.unwrapped.np_random
        cfg = self.cfg
        model = self.unwrapped.model

        self._current_mass_scale = float(
            rng.uniform(cfg.mass_scale_min, cfg.mass_scale_max)
        )
        self._current_friction_scale = float(
            rng.uniform(cfg.friction_scale_min, cfg.friction_scale_max)
        )
        self._current_damping_scale = float(
            rng.uniform(cfg.damping_scale_min, cfg.damping_scale_max)
        )
        self._current_motor_scale = float(
            rng.uniform(cfg.motor_scale_min, cfg.motor_scale_max)
        )

        model.body_mass[self._torso_id] = (
            self._original_torso_mass * self._current_mass_scale
        )
        model.geom_friction[:] = (
            self._original_geom_friction * self._current_friction_scale
        )
        model.dof_damping[:] = (
            self._original_dof_damping * self._current_damping_scale
        )
        model.actuator_gainprm[:] = (
            self._original_actuator_gainprm * self._current_motor_scale
        )

    def _restore_originals(self) -> None:
        model = self.unwrapped.model
        model.body_mass[self._torso_id] = self._original_torso_mass
        model.geom_friction[:] = self._original_geom_friction
        model.dof_damping[:] = self._original_dof_damping
        model.actuator_gainprm[:] = self._original_actuator_gainprm
        self._current_mass_scale = 1.0
        self._current_friction_scale = 1.0
        self._current_damping_scale = 1.0
        self._current_motor_scale = 1.0
