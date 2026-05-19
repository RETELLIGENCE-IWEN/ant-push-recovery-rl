from __future__ import annotations

import gymnasium as gym
import numpy as np


def main() -> None:
    seed = 42

    env = gym.make("Ant-v5")
    obs, info = env.reset(seed=seed)

    print("[env] Ant-v5 created")
    print("[obs] shape:", obs.shape)
    print("[action_space]:", env.action_space)
    print("[observation_space]:", env.observation_space)

    rng = np.random.default_rng(seed)

    total_reward = 0.0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        if terminated or truncated:
            print(f"[episode end] step={step}, terminated={terminated}, truncated={truncated}")
            obs, info = env.reset()

    env.close()
    print("[smoke test] success")
    print("[smoke test] total_reward:", total_reward)


if __name__ == "__main__":
    main()