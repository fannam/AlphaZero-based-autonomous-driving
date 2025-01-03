import numpy as np
import copy
from utils import init_stack_of_planes, get_stack_of_planes

class MCTSNode:
    def __init__(self, env, parent, parent_action, prior_prob, c_puct=2.5):
        self.env = copy.deepcopy(env)
        self.parent = parent #parent node
        self.parent_action = parent_action
        self.children = {} #parent.children[action] = child
        self._n = 0
        self._W = 0
        self._P = prior_prob #Xác suất thực hiện hành động parent_action tại parent_node
        self.c_puct = c_puct
        min_speed = self.env.unwrapped.road.vehicles[0].target_speeds[0]
        max_speed = self.env.unwrapped.road.vehicles[0].target_speeds[-1]
        self.speed_bonus = (self.env.unwrapped.road.vehicles[0].speed - min_speed)/(max_speed - min_speed)
        self.collision = 0
        self.brake_penalty = 0
        if self.parent_action==4:
            self.brake_penalty = 1
        if self.env.unwrapped.road.vehicles[0].crashed:
            self.collision = 1 + 2*self.speed_bonus
        if self.parent is None:
            self.stack_of_planes = init_stack_of_planes(self.env)
        else:
            self.stack_of_planes = get_stack_of_planes(self.env, self.parent.stack_of_planes)
    def pucb_score(self):
        """
        Tính PUCB của node
        """
        if self._n == 0:
            Q = 0
        else:
            Q = self._W / self._n

        return Q + self.c_puct * self._P * np.sqrt(self.parent._n) / (1 + self._n) + 0.5*self.speed_bonus - 0.4*self.collision - 0.2*self.brake_penalty
    def select(self):
        """
        Chọn node có UCB lớn nhất
        """
        if not self.children:  # Nếu không có node con
            return None  # Hoặc raise Exception("No children nodes to select from")
        return max(self.children.values(), key=lambda child: child.pucb_score())

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