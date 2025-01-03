import numpy as np
import copy
from stack_of_planes import init_stack_of_grid, get_stack_of_grid
import copy
import torch
from policy_smoothing import softmax_policy

grid_size=(21,5)
ego_position=(4,2)

class MCTSNode:
    def __init__(self, env, parent, parent_action, prior_prob):
        self.env = copy.deepcopy(env)
        self.parent = parent #parent node
        self.parent_action = parent_action
        self.children = {} #parent.children[action] = child
        self._n = 0
        self._W = 0
        self._P = prior_prob #Xác suất thực hiện hành động parent_action tại parent_node
        min_speed = self.env.unwrapped.road.vehicles[0].target_speeds[0]
        max_speed = self.env.unwrapped.road.vehicles[0].target_speeds[-1]
        self.speed_bonus = (self.env.unwrapped.road.vehicles[0].speed - min_speed)/(max_speed - min_speed)
        if self.env.unwrapped.road.vehicles[0].crashed:
            self.collision = 1
        else:
            self.collision = 0
        if self.parent is None:
            self.stack_of_planes = init_stack_of_grid(grid_size, ego_position)
        else:
            self.stack_of_planes = get_stack_of_grid(self.parent.stack_of_planes, env.unwrapped.observation_type.observe())
    def pucb_score(self, c_puct=2):
        """
        Tính PUCB của node
        """
        if self._n == 0:
            Q = 0
        else:
            Q = self._W / self._n

        return Q + c_puct * self._P * np.sqrt(np.log(self.parent._n) / (1 + self._n)) + 0.5*self.speed_bonus - 0.5*self.collision
    def select(self, c_puct=3):
        """
        Chọn node có UCB lớn nhất
        """
        if not self.children:  # Nếu không có node con
            return None  # Hoặc raise Exception("No children nodes to select from")
        return max(self.children.values(), key=lambda child: child.pucb_score(c_puct))

    def expand(self, action_priors):
        """
        Mở rộng cây bằng cách tạo node con
        action_priors là một dictionary chứa các xác suất prior của các action
        """
        for action, prob in action_priors.items():
            if action not in self.children and prob>0:
                #print(f"expanded {action} with {prob}")
                copy_env = copy.deepcopy(self.env)
                copy_env.step(action)
                self.children[action] = MCTSNode(copy_env, self, action, prob)

    def is_leaf(self):
        """
        Kiểm tra node có phải là leaf không
        """
        return self.children == {}
    def backpropagate(self, result):
        """
        Cập nhật visit count n và tổng điểm W
        new Q = new W/ new n
        """
        self._n += 1
        self._W += result
    def backpropagate_recursive(self, result):
        """
        Cập nhật toàn bộ đường đi từ node hiện tại đến root
        """
        if self.parent:
            self.parent.backpropagate_recursive(result)
        self.backpropagate(result)

class MCTS:
    def __init__(self, root, network, use_cuda=False, c_puct=5, n_simulations=10):
        self.c_puct = c_puct
        self.root = root
        if use_cuda:
            self._network = network.cuda()
        else:
            self._network = network
        self._n_simulations = n_simulations
    def traverse_to_leaf(self):
        node = self.root
        while not node.is_leaf():
            node = node.select(self.c_puct)
        return node

    def rollout(self):
        leaf_node = self.traverse_to_leaf()
        truncated = leaf_node.env.unwrapped._is_truncated() # True nếu hoàn thành episode
        crashed = leaf_node.env.unwrapped.road.vehicles[0].crashed # True nếu ego-vehicle xảy ra va chạm
        leaf_state = leaf_node.stack_of_planes
        state_tensor = torch.tensor(leaf_state, dtype=torch.float32).unsqueeze(0)
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
            predicted_value = 1.0
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