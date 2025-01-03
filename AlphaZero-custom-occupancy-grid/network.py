import torch
import torch.nn as nn
import torchvision.models as models

class ResNetRNNPolicyValue(nn.Module):
    def __init__(self, num_timesteps=5, num_actions=5):
        super(ResNetRNNPolicyValue, self).__init__()
        self.num_timesteps = num_timesteps

        # Load ResNet and modify for single-channel input
        resnet = models.resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Change input channels to 1
        resnet.fc = nn.Identity()  # Remove the fully connected layer

        self.resnet = resnet

        # Calculate output size after ResNet
        self.flatten_dim = 512  # Output of ResNet backbone

        # RNN Module
        self.rnn = nn.LSTM(input_size=self.flatten_dim, hidden_size=128, num_layers=1, batch_first=True)

        # Policy Head
        self.fc_policy = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
            nn.Softmax(dim=1),
        )

        # Value Head
        self.fc_value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        batch_size, timesteps, height, width = x.size()

        # Reshape to process each time step with ResNet
        x = x.view(batch_size * timesteps, 1, height, width)  # Add channel dimension (1)

        # Extract spatial features with ResNet
        x = self.resnet(x)
        x = x.view(batch_size, timesteps, -1)  # Flatten spatial dimensions

        # Learn temporal features with RNN
        x, _ = self.rnn(x)

        # Use the last time step output for predictions
        x_last = x[:, -1, :]

        # Policy and value predictions
        policy = self.fc_policy(x_last)
        value = self.fc_value(x_last)

        return policy, value

# # Example usage
# if __name__ == "__main__":
#     model = ResNetRNNPolicyValue(num_timesteps=5, num_actions=5)
#     sample_input = torch.randn(8, 5, 120, 20)  # Batch size 8, 5 timesteps, grid size 120x20
#     policy, value = model(sample_input)
#     print("Policy shape:", policy.shape)  # Expected: [8, 5]
#     print("Value shape:", value.shape)    # Expected: [8, 1]
