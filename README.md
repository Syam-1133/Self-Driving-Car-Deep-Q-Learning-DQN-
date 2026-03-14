<div align="center">

# Self-Driving Car AI
### Deep Reinforcement Learning on a Formula-Style Circuit

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Pygame](https://img.shields.io/badge/Pygame-2.0%2B-00B140?style=for-the-badge&logo=python&logoColor=white)](https://www.pygame.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<br/>

> An AI agent that teaches itself to drive a racing car around a Formula-style circuit — **from zero knowledge** — using only sensor data and reward signals. No hand-coded rules. No human demonstrations.

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-Syam--1133-181717?style=flat-square&logo=github)](https://github.com/Syam-1133)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-syam1133-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/syam1133/)
[![Email](https://img.shields.io/badge/Email-syamkklr123%40gmail.com-EA4335?style=flat-square&logo=gmail)](mailto:syamkklr123@gmail.com)

</div>

---

## What Is This?

This project applies **Deep Q-Learning (DQN)** to a custom 2D racing simulation. A car navigates a Formula Grand Prix circuit using only 7 rangefinder beams — no map, no GPS, no track data. The AI observes distances to the track walls, chooses a steering action, receives a reward, and gradually discovers how to drive through thousands of episodes of trial and error.

The result: an agent that goes from crashing in under a second to completing full laps — entirely self-taught.

---

## Key Features

- **Custom F1-style circuit** — hairpin, S-curves, chicane, long straights
- **7-beam LiDAR-style sensor system** — continuous distance readings, not binary
- **Deep Q-Network** — fully-connected neural net approximates the Q-function
- **Experience replay** — 100k transition buffer with random batch sampling
- **Live training dashboard** — real-time score plot as training runs
- **Clean modular codebase** — environment, agent, model all separated

---

## Demo

```
          ──── Sensor Fan ────
    -90° -60° -30°  0°  30° 60° 90°
      \    \    \   |   /   /   /
       ─────────── CAR ───────────
                  ↓ moves forward
```

The car always moves forward at a fixed speed. The agent only controls **steering** — left, straight, or right — based on what the 7 sensors return each frame.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DQN Agent                            │
│                                                         │
│  Environment         Neural Network        Action       │
│  ┌─────────┐        ┌──────────────┐      ┌─────────┐  │
│  │ 7 sensor│──────▶ │  FC: 7→256  │      │Straight │  │
│  │ readings│        │  ReLU        │─────▶│Right    │  │
│  │ (float) │        │  FC: 256→3  │      │Left     │  │
│  └─────────┘        └──────────────┘      └─────────┘  │
│                                                         │
│  Reward: +1 alive / -10 crash                           │
└─────────────────────────────────────────────────────────┘
```

### Neural Network

| Layer  | Size | Activation |
|--------|------|------------|
| Input  | 7    | —          |
| Hidden | 256  | ReLU       |
| Output | 3    | Linear     |

The 3 output neurons each hold the predicted Q-value for one action. The agent picks the action with the **highest Q-value**.

---

## How the Agent Learns

### 1 — Observe
The car reads 7 normalised sensor distances `[0.0 → 1.0]` representing how far each beam travels before hitting a wall.

### 2 — Act (Epsilon-Greedy)
```
epsilon = 80 - n_games

if random() < epsilon / 200:
    action = random choice          ← explore
else:
    action = argmax( Q(state) )     ← exploit
```
Early in training the agent explores randomly. After ~80 games it exploits learned knowledge exclusively.

### 3 — Reward
| Situation | Reward |
|-----------|--------|
| Alive on track | +1 per frame |
| Off-track crash | −10, episode ends |
| Timeout (1800 frames) | −1, episode ends |

### 4 — Learn (Bellman Update)
```
Q_target = reward + γ · max( Q(next_state) )     if not done
Q_target = reward                                 if done

Loss = MSE( Q_predicted, Q_target )
```
The network is trained to minimise the difference between what it predicted and what the Bellman equation says it should have predicted.

### 5 — Experience Replay
Every transition `(state, action, reward, next_state, done)` is stored in a **100,000-entry replay buffer**. After each episode, 1,000 random transitions are sampled and used for a batch update — breaking the correlation between consecutive samples and preventing catastrophic forgetting.

---

## Training Phases

| Phase | Episodes | What the Agent Does |
|-------|----------|---------------------|
| Random | 0 – 40 | Crashes almost instantly, steers randomly |
| Discovery | 40 – 150 | Stays on straights, begins recognising corners |
| Learning | 150 – 350 | Navigates most sections, occasional crashes at hairpin |
| Convergence | 350 – 600 | Completes full laps consistently |
| Mastery | 600+ | Smooth lap times, minimal crashes |

---

## The Track — Formula AI Grand Prix

A hand-designed multi-section circuit with eight distinct challenges:

```
              [Hairpin]
       ↗ [Back straight] ↗
  [S-curves]           [Turn 1]
      ↘                    ↘
  [Chicane]        [Main straight / Pit lane]
      ↘                    ↗
     [Slow left complex]
```

| Section | Description |
|---------|-------------|
| Main straight | Long flat-out run — car can go full speed |
| Turn 1 | Sweeping braking-zone right-hander |
| Back straight | Accelerating uphill section |
| Hairpin | Tight U-turn — most difficult for the AI |
| S-curves | Linked left-right-left esses |
| Chicane | Quick direction reversal |
| Slow left complex | Three-part corner sequence |
| Return chicane | Final flick back to the start/finish line |

The environment includes **grandstands with spectators**, **trackside trees**, **advertising boards**, and a **chequered start/finish line**.

---

## Project Structure

```
self-driving-car-dqn/
│
├── train.py                   ← Entry point — run this
├── requirements.txt           ← All dependencies
├── .gitignore
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── environment.py         ← Pygame simulation (track, sensors, rewards)
│   ├── agent.py               ← DQN agent (epsilon-greedy, replay, training)
│   ├── model.py               ← Neural network (Linear_QNet) + QTrainer
│   └── utils.py               ← Live training plot
│
└── checkpoints/
    └── car_model.pth          ← Best model saved automatically
```

---

## Getting Started

### Prerequisites
- Python 3.10 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Syam-1133/self-driving-car-dqn.git
cd self-driving-car-dqn

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Run Training

```bash
python train.py
```

Two windows open immediately:
- **Pygame window** — watch the car navigate the circuit in real time
- **Matplotlib window** — live chart of score per game and rolling average

The best-ever model is auto-saved to `checkpoints/car_model.pth` each time a new record is set.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pygame` | ≥ 2.0.0 | 2D simulation and rendering |
| `torch` | ≥ 1.9.0 | Neural network and backpropagation |
| `numpy` | ≥ 1.21.0 | State vector processing |
| `matplotlib` | ≥ 3.4.0 | Live training chart |
| `ipython` | ≥ 7.0.0 | Inline plot support |

---

## Tunable Parameters

| Parameter | File | Default | Effect |
|-----------|------|---------|--------|
| `CAR_SPEED` | `environment.py` | 3.0 | Forward speed (px/frame) |
| `TURN_SPEED` | `environment.py` | 4.5 | Steering rate (°/frame) |
| `NUM_SENSORS` | `environment.py` | 7 | Number of LiDAR beams |
| `SENSOR_MAX` | `environment.py` | 220 | Max sensor range (px) |
| `TIMEOUT` | `environment.py` | 1800 | Max frames per episode |
| `LR` | `agent.py` | 0.001 | Neural network learning rate |
| `GAMMA` | `agent.py` | 0.9 | Future reward discount factor |
| `MAX_MEMORY` | `agent.py` | 100,000 | Replay buffer capacity |
| `BATCH_SIZE` | `agent.py` | 1,000 | Training batch size |

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

## Author

**Syam Gudipudi**

*Passionate about AI, Reinforcement Learning, and building things that learn*

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-Syam--1133-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Syam-1133)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-syam1133-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/syam1133/)
[![Email](https://img.shields.io/badge/Email-syamkklr123%40gmail.com-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:syamkklr123@gmail.com)

<br/>

*If you found this project interesting, please consider giving it a ⭐*

</div>
