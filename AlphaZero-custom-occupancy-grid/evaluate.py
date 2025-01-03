from network import ResNetRNNPolicyValue
from CNN_network import AlphaZeroNetwork
from utils import softmax_policy, init_stack_of_planes, get_stack_of_planes
from trainer import AlphaZeroTrainer
from env_init import env_init
import torch
import numpy as np
import copy

ego_position=(4,2)
grid_size=(21,5)
model_path="alphazero_model (26).pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = ResNetRNNPolicyValue()
network = AlphaZeroNetwork(input_shape=(120, 20, 7))
number_of_step = []
average_speed = []

def evaluate(network, seed):
    action_list = []
    speed_list = []
    env = env_init(duration=40)
    state = init_stack_of_planes(env)
    trainer = AlphaZeroTrainer(network, env, c_puct=3.5, n_simulations=15, learning_rate=0.001, batch_size=64, epochs=30)
    trainer.load_model(model_path)
    trainer.network.eval()
    while not env.unwrapped._is_terminated() and not env.unwrapped._is_truncated():
        obs = env.unwrapped.observation_type.observe()
        state = get_stack_of_planes(env, state)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        predicted_policy, predicted_value = trainer.network(state_tensor)
        available_actions = env.unwrapped.get_available_actions()
        predicted_policy = {action: prob for action, prob in enumerate(predicted_policy.squeeze().tolist())}
        updated_policy = softmax_policy(predicted_policy, available_actions)
        action = max(updated_policy, key=updated_policy.get)
        print(updated_policy)
        env.render()
        env.step(action)
for seed in range(100, 120):
    evaluate(network=network ,seed=seed)

for i in range(len(average_speed)):
    print(f"steps: {number_of_step[i]}, avg speed: {average_speed[i]}")