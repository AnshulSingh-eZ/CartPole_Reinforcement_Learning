import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Env
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
rewards_data = []
# Q-network
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

qnet = QNet()
optimizer = optim.Adam(qnet.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Replay buffer
buffer = deque(maxlen=10000)

# Hyperparams
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
batch_size = 64

# Training
for episode in range(500):
    state, _ = env.reset()
    total_reward = 0

    while True:
        # Action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                qvals = qnet(torch.tensor(state, dtype=torch.float32))
                action = torch.argmax(qvals).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # Learn
        if len(buffer) >= batch_size:
            batch = random.sample(buffer, batch_size)
            s, a, r, ns, d = zip(*batch)

            s = torch.tensor(s, dtype=torch.float32)
            a = torch.tensor(a)
            r = torch.tensor(r, dtype=torch.float32)
            ns = torch.tensor(ns, dtype=torch.float32)
            d = torch.tensor(d, dtype=torch.float32)

            q = qnet(s).gather(1, a.unsqueeze(1)).squeeze()
            with torch.no_grad():
                q_next = qnet(ns).max(1)[0]
                target = r + gamma * q_next * (1 - d)

            loss = loss_fn(q, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_data.append(total_reward)
    print(f"Episode {episode}, Reward: {total_reward}")
torch.save(qnet.state_dict(), "cartpole_dqn.pth")
plt.figure(figsize=(10,5))
plt.plot(rewards_data, label="Episode Reward")
plt.show()

