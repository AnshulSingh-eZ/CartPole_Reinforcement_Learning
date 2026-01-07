import gymnasium as gym
import torch
import torch.nn as nn

# 1️⃣ Define the network (must match your trained one)
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x)

# 2️⃣ Load trained weights
qnet = QNet()
qnet.load_state_dict(torch.load("cartpole_dqn.pth"))
qnet.eval()  # important for inference

# 3️⃣ Create environment in human mode
env = gym.make("CartPole-v1", render_mode="human")
state, _ = env.reset(seed=None)

# 4️⃣ Run agent
while True:
    with torch.no_grad():
        # convert state to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        # choose action with max Q-value
        action = torch.argmax(qnet(state_tensor)).item()

    # take action
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    if done:
        state, _ = env.reset(seed=None)  # reset for next episode
