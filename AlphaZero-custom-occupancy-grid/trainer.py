import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from utils import init_stack_of_planes, get_stack_of_planes, softmax_policy
from MCTS import MCTS
from MCTS_node import MCTSNode
import torch
from env_init import env_init

duration = 15
env = env_init(duration=duration)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        state = init_stack_of_planes(env)
        done = self.env.unwrapped._is_truncated() or self.env.unwrapped._is_terminated()

        # Tạo gốc của cây MCTS
        root_node = MCTSNode(self.env, parent=None, parent_action=None, prior_prob=1.0, c_puct=self.c_puct)
        mcts = MCTS(root=root_node, network=self.network, c_puct=self.c_puct, n_simulations=self.n_simulations, duration=duration)

        while not done:
            # Thực hiện MCTS rollout để tính xác suất hành động
            state = get_stack_of_planes(env, state)
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


            # Chọn hành động dựa trên xác suất từ MCTS
            action = max(action_probs, key=action_probs.get)
            self.action_list.append(action)
            self.env.step(action)
            #print(f"action chosen: {action}")
            self.training_data.append((state_tensor, action_probs, predicted_value, action))
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
        self.network.to(device)
        states, policies, values, actions = zip(*self.training_data)
        states = torch.cat(states).to(device)
        policies = torch.tensor([list(policy.values()) for policy in policies], dtype=torch.float32).to(device)
        values = torch.tensor(values, dtype=torch.float32).unsqueeze(1).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)

        # Weighted sampling
        class_counts = torch.bincount(actions)
        class_weights = 1.0 / (class_counts.float() + 1e-6)
        sample_weights = class_weights[actions]

        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        dataset = TensorDataset(states, policies, values, actions)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)

        # Khởi tạo các danh sách lưu trữ loss
        self.policy_losses = []
        self.value_losses = []
        self.total_losses = []

        for epoch in range(self.epochs):
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_total_loss = 0
            batch_count = 0

            for state_batch, policy_batch, value_batch, action_batch in dataloader:
                # Move batches to device
                state_batch = state_batch.to(device)
                policy_batch = policy_batch.to(device)
                value_batch = value_batch.to(device)

                # Forward pass
                predicted_policy, predicted_value = self.network(state_batch)
                # Losses
                # policy_loss = F.kl_div(
                #     F.log_softmax(predicted_policy, dim=-1),
                #     policy_batch,
                #     reduction='batchmean'
                # )
                policy_loss = F.cross_entropy(predicted_policy, action_batch)
                value_loss = F.mse_loss(predicted_value, value_batch)
                loss = 0.9*policy_loss + 0.1*value_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Cộng dồn loss
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_total_loss += loss.item()
                batch_count += 1

            # Tính loss trung bình cho mỗi epoch
            avg_policy_loss = epoch_policy_loss / batch_count
            avg_value_loss = epoch_value_loss / batch_count
            avg_total_loss = epoch_total_loss / batch_count

            # Lưu loss vào danh sách
            self.policy_losses.append(avg_policy_loss)
            self.value_losses.append(avg_value_loss)
            self.total_losses.append(avg_total_loss)

            print(f"Epoch {epoch + 1}/{self.epochs}, value loss: {avg_value_loss}, policy loss: {avg_policy_loss}, Loss: {avg_total_loss}")



    def save_model(self, path="alphazero_model.pth"):
        torch.save(self.network.state_dict(), path)

    def load_model(self, path="alphazero_model.pth"):
        self.network.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
