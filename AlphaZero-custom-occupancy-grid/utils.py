import gymnasium as gym
import highway_env
from kinematics_to_occupancygrid import KinematicToGridWrapper
import numpy as np

def init_stack_of_planes(env, history_length=5):
    """
    Initialize a stack of planes representing the grid and velocity information.

    Args:
        env: The environment instance, expected to have the method `unwrapped.observation_type.observe()`
             to return the observation.

    Returns:
        A numpy array of shape (history_length, grid_height, grid_width) representing the stack of planes.
    """
    # Initialize the KinematicToGridWrapper
    converter = KinematicToGridWrapper()

    # Collect 5 observations and process each to form 15 planes
    grid_planes_list = []
    for _ in range(history_length):
        obs = env.unwrapped.observation_type.observe()
        grid_planes = converter.process_observation(obs, -2, 4 * (4 - 1) + 2)
        grid_planes_list.append(grid_planes)

    # Stack all 5 planes from the 5 observations
    grid_planes_stack = np.vstack(grid_planes_list)

    # Get ego-vehicle state from the latest observation
    # ego_x, ego_y, ego_vx, ego_vy, ego_heading = obs[0]
    # ego_speed = env.unwrapped.road.vehicles[0].speed

    # # Define max_speed and min_speed
    # max_speed = env.unwrapped.road.vehicles[0].target_speeds[-1]
    # min_speed = env.unwrapped.road.vehicles[0].target_speeds[0]

    # # Create the last 3 planes
    # grid_shape = grid_planes_stack.shape[1:]  # Extract grid height and width from grid_planes_stack

    # relative_speed_max_plane = np.full(grid_shape, ego_speed / max_speed, dtype=np.float32)
    # relative_speed_min_plane = np.full(grid_shape, ego_speed / max(min_speed, 1e-5), dtype=np.float32)  # Avoid division by zero
    # absolute_speed_plane = np.full(grid_shape, ego_speed, dtype=np.float32)

    # Stack all planes together
    stack_of_planes = np.vstack([
        grid_planes_stack,
        # relative_speed_max_plane[np.newaxis, :],
        # relative_speed_min_plane[np.newaxis, :],
        # absolute_speed_plane[np.newaxis, :]
    ])

    return stack_of_planes
import numpy as np

def get_stack_of_planes(env, old_state, history_length=5):
    """
    Update the stack of planes based on the latest observation.

    Args:
        env: The environment instance, expected to have the method `unwrapped.observation_type.observe()`
             to return the observation.
        old_state: A numpy array of shape (18, grid_height, grid_width) representing the previous state.

    Returns:
        A numpy array of shape (18, grid_height, grid_width) representing the updated stack of planes.
    """
    # Initialize the KinematicToGridWrapper
    converter = KinematicToGridWrapper()

    # Get new observation and process it to form 3 planes
    new_obs = converter.process_observation(env.unwrapped.observation_type.observe(), -2, 4 * (4 - 1) + 2)

    # Remove the first 3 planes (FIFO rule)
    updated_stack = old_state[1:history_length]
    #grid_shape = updated_stack.shape[1:]
    # Append the new observation planes to the stack
    updated_stack = np.vstack([updated_stack, new_obs])

    # Get ego-vehicle speed and speed limits
    # max_speed = env.unwrapped.road.vehicles[0].target_speeds[-1]
    # min_speed = env.unwrapped.road.vehicles[0].target_speeds[0]
    # ego_speed = env.unwrapped.road.vehicles[0].speed
    # relative_speed_max_plane = np.full(grid_shape, ego_speed / max_speed, dtype=np.float32)
    # relative_speed_min_plane = np.full(grid_shape, ego_speed / max(min_speed, 1e-5), dtype=np.float32)  # Avoid division by zero
    # absolute_speed_plane = np.full(grid_shape, ego_speed, dtype=np.float32)
    stack_of_planes = np.vstack([
        updated_stack,
        # relative_speed_max_plane[np.newaxis, :],
        # relative_speed_min_plane[np.newaxis, :],
        # absolute_speed_plane[np.newaxis, :]
    ])
    return stack_of_planes

import numpy as np

def softmax_policy(policy, available_actions):
    """
    Áp dụng softmax cho các xác suất trong policy dựa trên available_actions.

    :param policy: Dictionary chứa 5 hành động [0, 1, 2, 3, 4] với các xác suất tương ứng.
    :param available_actions: Danh sách các hành động có thể thực hiện (subset của [0, 1, 2, 3, 4]).
    :return: Dictionary chứa xác suất mới cho từng hành động (softmax áp dụng với các hành động khả dụng).
    """
    # Lấy các giá trị xác suất tương ứng với available_actions
    available_probs = np.array([policy[action] for action in available_actions])

    # Áp dụng softmax chỉ trên available_probs
    softmax_probs = available_probs / np.sum(available_probs)

    # Cập nhật xác suất mới
    updated_policy = {action: 0.0 for action in policy}  # Khởi tạo tất cả xác suất bằng 0
    for action, prob in zip(available_actions, softmax_probs):
        updated_policy[action] = prob

    return updated_policy