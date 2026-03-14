"""
src/agent.py  –  Deep Q-Learning agent for the self-driving car environment.
"""

import torch
import random
import numpy as np
from collections import deque

from .environment import CarGameAI, STATE_SIZE
from .model        import Linear_QNet, QTrainer
from .utils        import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 2_000    # larger batch → smoother gradient estimates
LR         = 0.0002   # lower LR → avoids weight overshooting in exploitation phase


class CarAgent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma   = 0.90       # slightly shorter horizon → more stable value estimates
        self.memory  = deque(maxlen=MAX_MEMORY)

        # 11 inputs → 256 hidden → 3 actions
        self.model   = Linear_QNet(STATE_SIZE, 256, 3)
        self.model.load("car_model.pth")   # resume from saved brain if it exists
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    # ── State ──────────────────────────────────────────────────────────────────
    def get_state(self, game: CarGameAI) -> np.ndarray:
        return game.get_state()   # (NUM_SENSORS,) float32, values in [0, 1]

    # ── Memory ─────────────────────────────────────────────────────────────────
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        batch = (random.sample(self.memory, BATCH_SIZE)
                 if len(self.memory) > BATCH_SIZE
                 else list(self.memory))
        states, actions, rewards, next_states, dones = zip(*batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    # ── Action selection ───────────────────────────────────────────────────────
    def get_action(self, state: np.ndarray):
        # Decay exploration over 400 games, floor at ~8% — gives enough time to learn
        self.epsilon = max(30, 400 - self.n_games)
        move = [0, 0, 0]
        if random.randint(0, 400) < self.epsilon:
            move[random.randint(0, 2)] = 1
        else:
            pred = self.model(torch.tensor(state, dtype=torch.float))
            move[torch.argmax(pred).item()] = 1
        return move


# ── Training loop ──────────────────────────────────────────────────────────────
def train():
    plot_scores      = []
    plot_mean_scores = []
    total_score      = 0
    record           = 0

    agent = CarAgent()
    game  = CarGameAI()

    while True:
        state_old = agent.get_state(game)
        action    = agent.get_action(state_old)

        reward, done, score = game.play_step(action)

        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save("car_model.pth")   # save best brain

            # Also checkpoint every 25 games so no progress is ever lost
            if agent.n_games % 25 == 0:
                agent.model.save("car_model_checkpoint.pth")

            print(f"Game {agent.n_games:4d} | Score {score:6d} | Record {record:6d}")

            plot_scores.append(score)
            total_score += score
            plot_mean_scores.append(total_score / agent.n_games)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()
