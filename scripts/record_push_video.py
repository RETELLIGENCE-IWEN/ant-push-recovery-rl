"""Record video of policy with a single externally-applied push at given time."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import gymnasium as gym
import imageio
import numpy as np
import torch
from stable_baselines3 import PPO

from stable_directional_ant import (
    WellTrainedLocomotionAntWrapper,
    WellTrainedLocomotionRewardConfig,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--push-step", type=int, default=200)
    parser.add_argument("--push-duration", type=int, default=5)
    parser.add_argument("--push-force", type=float, default=10.0)
    parser.add_argument("--push-direction-deg", type=float, default=90.0)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    torch.set_num_threads(1)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make("Ant-v5", render_mode="rgb_array")
    env = WellTrainedLocomotionAntWrapper(
        env, reward_config=WellTrainedLocomotionRewardConfig()
    )

    model = env.unwrapped.model
    torso_id = None
    for i in range(model.nbody):
        if model.body(i).name == "torso":
            torso_id = i
            break

    obs, info = env.reset(seed=args.seed)
    policy = PPO.load(args.model, device="cpu")

    theta = math.radians(args.push_direction_deg)
    force_xy = np.array([
        args.push_force * math.cos(theta),
        args.push_force * math.sin(theta),
    ])

    with imageio.get_writer(str(out_path), fps=args.fps) as writer:
        writer.append_data(env.render())
        for t in range(args.max_steps):
            env.unwrapped.data.xfrc_applied[torso_id, :] = 0.0
            if args.push_step <= t < args.push_step + args.push_duration:
                env.unwrapped.data.xfrc_applied[torso_id, 0] = force_xy[0]
                env.unwrapped.data.xfrc_applied[torso_id, 1] = force_xy[1]
            action, _ = policy.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            writer.append_data(env.render())
            if term or trunc:
                break
    env.close()
    print(f"[done] saved: {out_path}")


if __name__ == "__main__":
    main()
