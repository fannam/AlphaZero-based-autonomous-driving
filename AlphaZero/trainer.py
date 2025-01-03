import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from policy_smoothing import softmax_policy
from stack_of_planes import init_stack_of_grid, get_stack_of_grid
from MCTS import MCTSNode, MCTS

grid_size=(21,5)
ego_position=(4,2)

class AlphaZeroTrainer:
    def __init__(self, network, env, c_puct=2, n_simulations=10, learning_rate=0.001, batch_size=32, epochs=10):
        self.network = network  # AlphaZeroNetwork
        self.env = env
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.epochs = epochs
        self.training_data = []  # Lưu trữ dữ liệu huấn luyện dạng (state, policy, value)
        self.action_list = []

    def self_play(self, seed=21):
        """
        Tạo dữ liệu huấn luyện thông qua self-play với MCTS.
        """
        # Khởi tạo lại môi trường và trạng thái ban đầu
        self.env.reset(seed=seed)
        state = init_stack_of_grid(grid_size, ego_position)
        done = self.env.unwrapped._is_truncated() or self.env.unwrapped._is_terminated()

        # Tạo gốc của cây MCTS
        root_node = MCTSNode(self.env, parent=None, parent_action=None, prior_prob=1.0)
        mcts = MCTS(root=root_node, network=self.network, use_cuda=False, c_puct=self.c_puct, n_simulations=self.n_simulations)

        while not done:
            # Thực hiện MCTS rollout để tính xác suất hành động
            state = get_stack_of_grid(state, self.env.unwrapped.observation_type.observe())
            for _ in range(self.n_simulations):
                mcts.rollout()
            # Thu thập xác suất hành động và giá trị của trạng thái hiện tại
            action_probs = {action: 0.0 for action in range(5)}  # Khởi tạo xác suất của tất cả hành động là 0
            for action, child in root_node.children.items():
                action_probs[action] = child._n / (root_node._n - 1)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            #print(state_tensor.shape)
            predicted_value = root_node._W / root_node._n if root_node._n > 0 else 0

            # Lưu trữ dữ liệu huấn luyện
            self.training_data.append((state_tensor, action_probs, predicted_value))

            # Chọn hành động dựa trên xác suất từ MCTS
            action = max(action_probs, key=action_probs.get)
            self.action_list.append(action)
            self.env.step(action)
            print(f"action chosen: {action}")
            #(env.unwrapped.road.vehicles[0].target_lane_index[2])

            # Di chuyển gốc của MCTS đến node con tương ứng với hành động được chọn
            if action in root_node.children:
                mcts.move_to_new_root(action)
                root_node = mcts.root  # Cập nhật root_node cho vòng lặp kế tiếp
            else:
                raise ValueError("Action không tồn tại trong cây MCTS.")

            # Cập nhật trạng thái và kiểm tra điều kiện kết thúc
            done = self.env.unwrapped._is_truncated() or self.env.unwrapped._is_terminated()
        print("end self-play")

    def train(self):
        """
        Huấn luyện mạng neural trên dữ liệu self-play đã thu thập được.
        """
        # Tách dữ liệu thành tensor cho huấn luyện
        states, policies, values = zip(*self.training_data)
        states = torch.cat(states)
        #print(states.shape)
        policies = torch.tensor([list(policy.values()) for policy in policies], dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32).unsqueeze(1)

        # Tạo DataLoader để huấn luyện theo batch
        dataset = TensorDataset(states, policies, values)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for state_batch, policy_batch, value_batch in dataloader:
                # Forward pass
                predicted_policy, predicted_value = self.network(state_batch)
                policy_loss = F.cross_entropy(predicted_policy, policy_batch)
                value_loss = F.mse_loss(predicted_value, value_batch)
                loss = policy_loss + value_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item()}")

    def save_model(self, path="alphazero_model.pth"):
        torch.save(self.network.state_dict(), path)

    def load_model(self, path="alphazero_model.pth"):
        self.network.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    
