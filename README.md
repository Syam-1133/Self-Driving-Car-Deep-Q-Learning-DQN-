# Self-Driving Car — Deep Q-Learning (DQN)

A reinforcement learning simulation where an AI agent learns to drive a car around a Formula-style racing circuit entirely on its own — no human input, no hard-coded rules. The agent starts with zero knowledge and gradually learns to stay on track through trial, error, and reward signals.

**Author:** Syam Gudipudi
[![GitHub](https://img.shields.io/badge/GitHub-Syam--1133-181717?logo=github)](https://github.com/Syam-1133)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-syam1133-0A66C2?logo=linkedin)](https://www.linkedin.com/in/syam1133/)
[![Email](https://img.shields.io/badge/Email-syamkklr123@gmail.com-EA4335?logo=gmail)](mailto:syamkklr123@gmail.com)

---

## Demo

![Track Preview](https://i.imgur.com/placeholder.png)

> The car (red) casts 7 sensor rays to perceive the track boundaries and learns to steer itself around the circuit.

---

## How It Works

### The Big Idea — Reinforcement Learning

The agent learns by interacting with the environment in a loop:

```
Observe state → Choose action → Get reward → Update knowledge → Repeat
```

Over thousands of episodes the agent figures out which steering decisions lead to staying on track longer, and which ones lead to crashes.

---

### Algorithm — Deep Q-Learning (DQN)

The agent uses **Deep Q-Network (DQN)**, a reinforcement learning algorithm that combines Q-Learning with a neural network.

**Q-Learning** maintains a value function `Q(state, action)` that estimates the total future reward of taking a given action in a given state. The agent always picks the action with the highest Q-value.

The update rule is the **Bellman Equation**:

```
Q(s, a)  ←  reward  +  γ · max Q(s', a')
```

| Symbol | Meaning |
|--------|---------|
| `s`    | Current state |
| `a`    | Action taken |
| `s'`   | Next state after action |
| `γ`    | Discount factor (0.9) — how much future rewards matter |

Because the state space is continuous (sensor distances), a **neural network** approximates the Q-function instead of a lookup table.

---

### Neural Network Architecture

```
Input Layer       Hidden Layer      Output Layer
  (7 neurons)  →  (256 neurons)  →  (3 neurons)
  sensor rays      ReLU activation   Q-values per action
```

| Layer  | Size | Activation |
|--------|------|------------|
| Input  | 7    | —          |
| Hidden | 256  | ReLU       |
| Output | 3    | Linear     |

The output gives a Q-value for each of the 3 possible actions. The agent picks the action with the highest Q-value.

---

### State Space — What the Car Sees

The car has **7 raycasting sensors** fanning out in front of it:

```
         -90°  -60°  -30°   0°   30°  60°  90°
           \     \     \    |    /    /    /
            ──────────── CAR ────────────
```

Each sensor shoots a ray in its direction and returns the **normalised distance** to the nearest track boundary (0 = wall right here, 1 = nothing within range). This gives the agent a picture of the space around it similar to how a real self-driving car uses LiDAR.

---

### Action Space — What the Car Can Do

The agent chooses one of 3 actions each frame:

| Action         | Encoding  | Effect            |
|----------------|-----------|-------------------|
| Go straight    | `[1,0,0]` | Keep current angle |
| Turn right     | `[0,1,0]` | Rotate +4.5°/frame |
| Turn left      | `[0,0,1]` | Rotate −4.5°/frame |

The car always moves forward at a constant speed. The agent only controls steering.

---

### Reward Function

| Event              | Reward |
|--------------------|--------|
| Alive on track     | +1 per frame |
| Off-track crash    | −10, episode ends |
| Timeout (1800 frames) | −1, episode ends |

The agent maximises cumulative reward, which means it learns that surviving longer = better.

---

### Training — Experience Replay

Two training mechanisms run in parallel to make learning stable:

| Type               | When              | Description |
|--------------------|-------------------|-------------|
| **Short memory**   | Every step        | Immediately trains on the single latest transition |
| **Long memory**    | End of episode    | Randomly samples 1000 transitions from a 100k replay buffer and trains on the batch |

Sampling random batches breaks the correlation between consecutive experiences, which prevents the network from forgetting old lessons while learning new ones.

---

### Exploration vs Exploitation

Early in training the agent needs to try random actions to discover what works. Later it should trust what it has learned. This is controlled by **epsilon-greedy** scheduling:

```
epsilon = 80 - n_games
P(random action) = epsilon / 200
```

- **Game 0:**  40% chance of random action
- **Game 80+:** 0% random — fully exploits learned policy

---

## Project Structure

```
Self-Driving-Car-DQN/
│
├── train.py               ← Entry point — run this to start training
├── requirements.txt       ← Python dependencies
├── .gitignore
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── environment.py     ← Pygame simulation (track, car, sensors, rewards)
│   ├── agent.py           ← DQN agent (epsilon-greedy, memory, training loop)
│   ├── model.py           ← Neural network (Linear_QNet) + QTrainer
│   └── utils.py           ← Live training plot
│
└── checkpoints/           ← Saved model weights (best score so far)
    └── car_model.pth
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Syam-1133/self-driving-car-dqn.git
cd self-driving-car-dqn
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Requirements:**

| Package      | Version  | Purpose                     |
|--------------|----------|-----------------------------|
| `pygame`     | ≥ 2.0.0  | Game simulation & rendering |
| `torch`      | ≥ 1.9.0  | Neural network (DQN)        |
| `numpy`      | ≥ 1.21.0 | Numerical state arrays      |
| `matplotlib` | ≥ 3.4.0  | Live training plot          |
| `ipython`    | ≥ 7.0.0  | Inline plot display         |

### 3. Train the agent

```bash
python train.py
```

A Pygame window opens showing the car navigating the track. A second window shows the live score chart. The best model is automatically saved to `checkpoints/car_model.pth` whenever a new high score is reached.

---

## Training Progress

What to expect as training runs:

| Phase        | Games     | Behaviour |
|--------------|-----------|-----------|
| **Random**   | 0 – 40    | Car crashes almost immediately, mostly random steering |
| **Learning** | 40 – 150  | Car starts staying on straights, struggles at corners |
| **Improving**| 150 – 500 | Car navigates most of the circuit, occasional crashes |
| **Converged**| 500+      | Car completes full laps consistently |

---

## The Track — Formula AI Grand Prix

The circuit is a hand-designed F1-style layout featuring:

| Section           | Description |
|-------------------|-------------|
| Main straight     | Long flat-out run with centre-line dashes |
| Turn 1            | Sweeping braking-zone right-hander |
| Back straight     | Accelerating uphill run |
| Top hairpin       | Tight U-turn — hardest section for the AI |
| S-curves / esses  | Linked left-right-left curves |
| Chicane           | Quick direction change |
| Slow left complex | Three-part corner on the left side |
| Return chicane    | Final flick back onto the main straight |

The environment also includes grandstands with spectators, trackside trees, advertising boards, and a chequered start/finish line.

---

## Key Parameters

All tunable constants are at the top of their respective files:

| Parameter     | File              | Default | Effect |
|---------------|-------------------|---------|--------|
| `CAR_SPEED`   | `environment.py`  | 3.0     | Forward speed (px/frame) |
| `TURN_SPEED`  | `environment.py`  | 4.5     | Steering rate (°/frame) |
| `SENSOR_MAX`  | `environment.py`  | 220     | Max sensor range (px) |
| `TIMEOUT`     | `environment.py`  | 1800    | Max frames per episode |
| `LR`          | `agent.py`        | 0.001   | Neural network learning rate |
| `GAMMA`       | `agent.py`        | 0.9     | Discount factor |
| `MAX_MEMORY`  | `agent.py`        | 100,000 | Replay buffer size |
| `BATCH_SIZE`  | `agent.py`        | 1,000   | Training batch size |

---

## Built With

- **Python 3.10+**
- **PyTorch** — neural network and backpropagation
- **Pygame** — real-time 2D simulation and rendering
- **NumPy** — state vector processing
- **Matplotlib** — live training visualisation

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Syam Gudipudi**

| | |
|---|---|
| GitHub   | [github.com/Syam-1133](https://github.com/Syam-1133) |
| LinkedIn | [linkedin.com/in/syam1133](https://www.linkedin.com/in/syam1133/) |
| Email    | [syamkklr123@gmail.com](mailto:syamkklr123@gmail.com) |
