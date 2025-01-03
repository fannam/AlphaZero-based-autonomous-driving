import copy
import torch
from utils import init_stack_of_planes, get_stack_of_planes, softmax_policy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
duration = 15

class MCTS:
    def __init__(self, root, network, c_puct=3.5, n_simulations=10, min_average_speed=23, duration=12):
        self.c_puct = c_puct
        self.root = root
        self._network = network.to(device)
        self._n_simulations = n_simulations
        self.ego_init_position = root.env.unwrapped.road.vehicles[0].position[0]
        self.min_average_speed = min_average_speed
        self.duration = duration
    def traverse_to_leaf(self):
        node = self.root
        while not node.is_leaf():
            node = node.select()
        return node

    def rollout(self):
        leaf_node = self.traverse_to_leaf()
        truncated = leaf_node.env.unwrapped._is_truncated() # True nếu hoàn thành episode
        crashed = leaf_node.env.unwrapped.road.vehicles[0].crashed # True nếu ego-vehicle xảy ra va chạm
        leaf_state = leaf_node.stack_of_planes
        state_tensor = torch.tensor(leaf_state, dtype=torch.float32).unsqueeze(0).to(device)
        predicted_policy, predicted_value = self._network(state_tensor)
        predicted_policy = {action: prob for action, prob in enumerate(predicted_policy.squeeze().tolist())}
        #print(predicted_policy)
        available_actions = leaf_node.env.unwrapped.get_available_actions()
        #print(available_actions)
        updated_policy = softmax_policy(predicted_policy, available_actions)
        #print(updated_policy)
        predicted_value = predicted_value.item()
        if not truncated and not crashed:
            leaf_node.expand(updated_policy)
        elif truncated:
            ego_last_position = leaf_node.env.unwrapped.road.vehicles[0].position[0]
            ego_average_speed = (ego_last_position - self.ego_init_position)/(self.duration-1)
            confidence_score = ego_average_speed / self.min_average_speed
            if confidence_score >=1.0:
                predicted_value = 1.0
            else:
                predicted_value = 0.0
        elif crashed:
            predicted_value = -1.0
        leaf_node.backpropagate_recursive(predicted_value)

    def move_to_new_root(self, action):
        """
        Chuyển gốc của cây MCTS tới node con tương ứng với hành động được chọn.
        """
        if action in self.root.children:
            self.root = self.root.children[action]  # Di chuyển gốc tới node con
            self.root.parent = None  # Ngắt liên kết với node cha để giảm bộ nhớ
        else:
            # Nếu node con không tồn tại, khởi tạo lại cây tại node gốc mới
            raise ValueError("Hành động không có trong cây hiện tại.")