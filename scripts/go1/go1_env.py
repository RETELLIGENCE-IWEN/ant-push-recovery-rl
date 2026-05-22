"""
Go1Env: a gym.Env wrapper around the MuJoCo Menagerie Unitree Go1 model.

M2 milestone — environment validation, not training. Action is a residual on
the home-pose joint targets; control runs through the model's built-in
position actuators (kp=100 internal PD) at policy rate after decimation from
the 500 Hz physics step.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


DEFAULT_MODEL_PATH = Path("/workspace/external/mujoco_menagerie/unitree_go1/scene.xml")
N_JOINTS = 12
OBS_DIM = 48  # 3+3+3 + 3 + 12+12+12


@dataclass(frozen=True)
class Go1EnvConfig:
    model_path: Path = DEFAULT_MODEL_PATH

    # Control
    action_scale: float = 0.15           # rad of residual added to home_ctrl
    decimation: int = 10                  # physics steps per policy step (500Hz / 10 = 50Hz)

    # Settle on reset
    settle_steps: int = 100               # 100 phys steps @ dt=0.002 = 0.2s

    # Commands (constant for M2 / M3; later swapped for a sampler)
    command_vx: float = 0.0
    command_vy: float = 0.0
    command_wz: float = 0.0

    # Termination
    terminate_z_min: float = 0.16
    terminate_tilt_max: float = 0.8       # rad on roll or pitch

    # Episode length (counted in policy steps)
    max_episode_steps: int = 1000


class Go1Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        config: Go1EnvConfig | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.cfg = config or Go1EnvConfig()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(self.cfg.model_path))
        self.data = mujoco.MjData(self.model)

        # Locate home keyframe + torso body
        self.home_key_id = next(
            (i for i in range(self.model.nkey) if self.model.key(i).name == "home"),
            -1,
        )
        if self.home_key_id < 0:
            raise RuntimeError("Go1 model is missing the `home` keyframe.")
        self.home_qpos = np.array(self.model.key_qpos[self.home_key_id], copy=True)
        self.home_ctrl = np.array(self.model.key_ctrl[self.home_key_id], copy=True)

        self.trunk_body_id = next(
            i for i in range(self.model.nbody) if self.model.body(i).name == "trunk"
        )
        self.actuator_ctrl_min = np.array(self.model.actuator_ctrlrange[:, 0], copy=True)
        self.actuator_ctrl_max = np.array(self.model.actuator_ctrlrange[:, 1], copy=True)

        # Spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(N_JOINTS,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        # Internal state
        self._prev_action = np.zeros(N_JOINTS, dtype=np.float64)
        self._step_counter = 0
        self._renderer: mujoco.Renderer | None = None

    # ---------- gym interface ----------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.home_ctrl
        # Settle so contact forces are well-balanced before the policy takes over.
        for _ in range(self.cfg.settle_steps):
            mujoco.mj_step(self.model, self.data)
        self._prev_action = np.zeros(N_JOINTS, dtype=np.float64)
        self._step_counter = 0
        obs = self._compute_observation()
        return obs, self._build_info()

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        q_target = self.home_ctrl + self.cfg.action_scale * a
        ctrl = np.clip(q_target, self.actuator_ctrl_min, self.actuator_ctrl_max)

        for _ in range(self.cfg.decimation):
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

        self._step_counter += 1
        obs = self._compute_observation()
        terminated, term_reason = self._check_termination()
        truncated = self._step_counter >= self.cfg.max_episode_steps
        reward = 0.0  # M2: placeholder; M3+ supplies reward
        self._prev_action = a.copy()

        info = self._build_info()
        info["term_reason"] = term_reason
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="tracking")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ---------- internals ----------

    def _quat_to_rotmat(self, quat_wxyz: np.ndarray) -> np.ndarray:
        out = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(out, quat_wxyz)
        return out.reshape(3, 3)

    def _compute_observation(self) -> np.ndarray:
        quat = self.data.qpos[3:7]              # wxyz
        R = self._quat_to_rotmat(quat)          # body-to-world rotation
        Rt = R.T                                # world-to-body

        lin_vel_world = self.data.qvel[0:3]
        ang_vel_world = self.data.qvel[3:6]
        base_lin_vel_body = Rt @ lin_vel_world
        base_ang_vel_body = Rt @ ang_vel_world

        gravity_world = np.array([0.0, 0.0, -1.0])
        projected_gravity = Rt @ gravity_world

        command = np.array(
            [self.cfg.command_vx, self.cfg.command_vy, self.cfg.command_wz],
            dtype=np.float64,
        )
        joint_pos_error = self.data.qpos[7:] - self.home_ctrl
        joint_vel = self.data.qvel[6:]

        obs = np.concatenate(
            [
                base_lin_vel_body,
                base_ang_vel_body,
                projected_gravity,
                command,
                joint_pos_error,
                joint_vel,
                self._prev_action,
            ]
        ).astype(np.float32)
        return obs

    def _check_termination(self) -> tuple[bool, str]:
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            return True, "nan"
        z = float(self.data.qpos[2])
        if z < self.cfg.terminate_z_min:
            return True, f"z<{self.cfg.terminate_z_min}"
        # roll/pitch via quat
        quat = self.data.qpos[3:7]
        w, x, y, zq = [float(v) for v in quat]
        import math
        sinr_cosp = 2 * (w * x + y * zq)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = max(-1.0, min(1.0, 2 * (w * y - zq * x)))
        pitch = math.asin(sinp)
        if abs(roll) > self.cfg.terminate_tilt_max:
            return True, f"|roll|>{self.cfg.terminate_tilt_max}"
        if abs(pitch) > self.cfg.terminate_tilt_max:
            return True, f"|pitch|>{self.cfg.terminate_tilt_max}"
        return False, ""

    def _build_info(self) -> dict:
        z = float(self.data.qpos[2])
        return {
            "z": z,
            "qpos7": [float(x) for x in self.data.qpos[:7]],
            "joint_pos": [float(x) for x in self.data.qpos[7:]],
            "joint_vel": [float(x) for x in self.data.qvel[6:]],
            "step": self._step_counter,
            "sim_time_s": float(self.data.time),
        }
