import gymnasium as gym
import highway_env
import numpy as np

def env_init(duration):
    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 8,
            "features": ["x", "y", "vx", "vy", "heading"],
            "absolute": True,
            "normalize": False,
            "order": "sorted",
        },
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": np.linspace(20, 30, 3)
        },
        "lanes_count": 4,
        "vehicles_density": 1.0,
        "duration": duration,
    }
    env = gym.make("highway-v0", config=config, render_mode='rgb_array')
    env.reset()
    return env