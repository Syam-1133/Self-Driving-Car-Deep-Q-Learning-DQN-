import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './checkpoints'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        torch.save(self.state_dict(), os.path.join(model_folder_path, file_name))

    def load(self, file_name='model.pth'):
        path = os.path.join('./checkpoints', file_name)
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, weights_only=True))
            print(f"[Brain] Loaded saved weights from {path}")
            return True
        return False


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr       = lr
        self.gamma    = gamma
        self.model    = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        # Target network — frozen copy updated every 200 steps.
        # Provides stable Q-value targets so weights don't chase a moving signal.
        self.target_model   = copy.deepcopy(model)
        self.target_model.eval()
        self._train_counter = 0
        self.TARGET_UPDATE  = 500   # sync target ← online every N train calls (slower = more stable)

    def train_step(self, state, action, reward, next_state, done):
        state      = torch.tensor(state,      dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action     = torch.tensor(action,     dtype=torch.long)
        reward     = torch.tensor(reward,     dtype=torch.float)

        if len(state.shape) == 1:
            state      = torch.unsqueeze(state,      0)
            next_state = torch.unsqueeze(next_state, 0)
            action     = torch.unsqueeze(action,     0)
            reward     = torch.unsqueeze(reward,     0)
            done       = (done, )

        # Predicted Q-values from the online network
        pred   = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Use frozen target network for next-state value → stable targets
                with torch.no_grad():
                    Q_new = reward[idx] + self.gamma * torch.max(
                        self.target_model(next_state[idx])
                    )
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        # Gradient clipping — stops weight explosion that caused the score collapse
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Periodically copy online weights → target network
        self._train_counter += 1
        if self._train_counter % self.TARGET_UPDATE == 0:
            self.target_model.load_state_dict(self.model.state_dict())



