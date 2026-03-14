<div align="center">

# Self-Driving Car AI
### Deep Reinforcement Learning on a Formula-Style Circuit

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Pygame](https://img.shields.io/badge/Pygame-2.0%2B-00B140?style=for-the-badge&logo=python&logoColor=white)](https://www.pygame.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<br/>

> An AI agent that teaches itself to drive a Formula-style racing car around a full circuit — **from zero knowledge** — using only sensor data and reward signals. No hand-coded rules. No human demonstrations. Just trial, error, and learning.

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-Syam--1133-181717?style=flat-square&logo=github)](https://github.com/Syam-1133)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-syam1133-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/syam1133/)
[![Email](https://img.shields.io/badge/Email-syamkklr123%40gmail.com-EA4335?style=flat-square&logo=gmail)](mailto:syamkklr123@gmail.com)

</div>

---

## What Is This?

This project applies **Deep Q-Learning (DQN)** to a custom 2D racing simulation built with Pygame. A car navigates a Formula Grand Prix circuit using 9 LiDAR-style rangefinder beams — no map, no GPS, no track data. The AI observes wall distances and its angle/distance to the next checkpoint, chooses a steering action, receives a shaped reward signal, and gradually discovers how to drive through thousands of episodes of trial and error.

The result: an agent that goes from crashing in under a second to completing full laps — entirely self-taught.

---

## Key Features

- **Custom F1-style circuit** — hairpin, S-curves, chicane, long straights, 8 checkpoint gates
- **9-beam LiDAR sensor system** — wide angular coverage with denser forward arc for early corner detection
- **11-input state vector** — 9 sensors + angle-to-checkpoint + distance-to-checkpoint
- **Deep Q-Network with Target Network** — frozen copy updated every 500 steps for stable Q-value targets
- **Dense reward shaping** — progress reward, checkpoint gates, lap bonus, wall proximity penalty
- **Persistent brain** — model auto-saves every 25 games and on new records; resumes on restart
- **Live training dashboard** — real-time score plot, rolling average, and stats panel
- **Gradient clipping** — prevents weight explosion during training
- **Clean modular codebase** — environment, agent, model, utils all separated

---

## Demo

```
          ──── Sensor Fan (9 beams) ────
  -90° -60° -30° -15°  0°  15° 30° 60° 90°
     \    \    \    \   |   /   /   /   /
      ──────────────── CAR ────────────────
                       ↓ moves forward
```

The car always moves forward at a fixed speed. The agent only controls **steering** — left, straight, or right — based on what the 9 sensors return each frame plus the direction and distance to the next checkpoint.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        DQN Agent                            │
│                                                             │
│  State (11 inputs)      Neural Network        Action        │
│  ┌──────────────┐       ┌─────────────┐      ┌──────────┐  │
│  │ 9 sensor     │       │ FC: 11→256  │      │ Straight │  │
│  │ readings     │──────▶│ ReLU        │─────▶│ Right    │  │
│  │ angle to CP  │       │ FC: 256→3   │      │ Left     │  │
│  │ dist to CP   │       └─────────────┘      └──────────┘  │
│  └──────────────┘                                           │
│                                                             │
│  Online Network ──trains──▶ every step                      │
│  Target Network ──syncs──▶ every 500 train calls            │
└─────────────────────────────────────────────────────────────┘
```

### Neural Network

| Layer  | Size | Activation |
|--------|------|------------|
| Input  | 11   | —          |
| Hidden | 256  | ReLU       |
| Output | 3    | Linear     |

The 3 output neurons hold the predicted Q-value for each action. The agent picks the action with the **highest Q-value**.

---

## How the Agent Learns

### 1 — Observe
The car reads a state vector of **11 values**:
- 9 normalised sensor distances `[0.0 → 1.0]`
- Relative angle to next checkpoint `[-1.0 → +1.0]`
- Normalised distance to next checkpoint `[0.0 → 1.0]`

### 2 — Act (Epsilon-Greedy)
```
epsilon = max(30, 400 - n_games)

if random(0, 400) < epsilon:
    action = random choice          ← explore
else:
    action = argmax( Q(state) )     ← exploit
```
Exploration stays above **7.5% for all 400 games**, giving the agent enough time to discover the full circuit before committing to a policy.

### 3 — Reward Shaping

| Situation | Reward |
|-----------|--------|
| Alive, heading toward checkpoint | +1.0 to +3.0 per frame |
| Off-track crash | −10, episode ends |
| Wall proximity (< 30px) | up to −1.5 per frame |
| Checkpoint collected | +50 |
| Full lap completed | +200 bonus |
| Timeout (2000 frames) | −1, episode ends |

### 4 — Learn (Bellman Update with Target Network)
```
Q_target = reward + γ · max( Q_target_network(next_state) )   if not done
Q_target = reward                                               if done

Loss = MSE( Q_predicted, Q_target )
```
The **target network** is a frozen copy of the online network. It provides stable Q-value targets so the online network isn't chasing a constantly moving signal — a key fix for training stability.

### 5 — Experience Replay
Every transition `(state, action, reward, next_state, done)` is stored in a **100,000-entry replay buffer**. After each episode, **2,000 random transitions** are sampled for a batch update — breaking correlation between consecutive samples.

### 6 — Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
Prevents weight explosion that caused score collapse in early training iterations.

### 7 — Persistent Brain
The model saves automatically:
- **On every new record score** → `checkpoints/car_model.pth`
- **Every 25 games** → `checkpoints/car_model_checkpoint.pth`

On the next run, the agent loads the saved weights and **resumes from where it left off** — no progress is ever lost.

---

## Training Phases

| Phase | Episodes | What the Agent Does |
|-------|----------|---------------------|
| Random | 0 – 50 | Crashes almost instantly, steers randomly |
| Discovery | 50 – 150 | Stays on straights, begins recognising corners |
| Learning | 150 – 300 | Navigates most sections, hits checkpoints consistently |
| Convergence | 300 – 500 | Completes full laps, rare crashes |
| Mastery | 500+ | Smooth lap times, consistent circuit completion |

---

## The Track — Formula AI Grand Prix

A hand-designed multi-section circuit with 8 checkpoint gates and distinct driving challenges:

```
              [Hairpin apex]
       ↗ [Back straight] ↗
  [S-curves]           [Turn 1]
      ↘                    ↘
  [Chicane]        [Main straight / S/F line]
      ↘                    ↗
     [Slow left complex]
```

| Section | Description |
|---------|-------------|
| Main straight | Long flat-out run — car reaches full speed |
| Turn 1 | Sweeping braking-zone right-hander |
| Back straight | Accelerating section toward hairpin |
| Hairpin | Tight U-turn — most challenging for the AI |
| S-curves | Linked left-right esses |
| Chicane | Quick direction reversal |
| Slow left complex | Three-part corner sequence |
| Return chicane | Final flick back to start/finish |

The environment includes **grandstands with spectators**, **trackside trees**, **advertising hoardings**, **centre-line dashes**, and a **chequered start/finish line**.

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
│   ├── environment.py         ← Pygame simulation (track, sensors, rewards, rendering)
│   ├── agent.py               ← DQN agent (epsilon-greedy, replay, training loop)
│   ├── model.py               ← Neural network + QTrainer + target network
│   └── utils.py               ← Live training plot (persistent figure, dark theme)
│
└── checkpoints/
    ├── car_model.pth           ← Best model (saved on every new record)
    └── car_model_checkpoint.pth ← Periodic checkpoint (saved every 25 games)
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
- **Matplotlib window** — live chart with score per game, 20-game rolling average, and stats panel

Training resumes automatically from the last saved checkpoint if one exists.

---

## Hyperparameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `LR` | 0.0002 | Low learning rate — stable weight updates |
| `GAMMA` | 0.90 | Future reward discount factor |
| `MAX_MEMORY` | 100,000 | Replay buffer capacity |
| `BATCH_SIZE` | 2,000 | Large batch — smoother gradient estimates |
| `epsilon` decay | 400 games | Slow exploration decay — avoids premature convergence |
| `epsilon` floor | 7.5% | Always some exploration — prevents policy collapse |
| `TARGET_UPDATE` | 500 steps | Target network sync frequency — stable Q-targets |
| `CAR_SPEED` | 4.0 px/frame | Forward speed |
| `TURN_SPEED` | 5.0 °/frame | Steering rate |
| `NUM_SENSORS` | 9 | LiDAR beams |
| `SENSOR_MAX` | 220 px | Max sensor range |
| `TIMEOUT` | 2000 frames | Max frames per episode |
| `CHECKPOINT_REWARD` | +50 | Reward per gate collected |
| `LAP_REWARD` | +200 | Bonus for completing a full lap |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pygame` | ≥ 2.0.0 | 2D simulation and rendering |
| `torch` | ≥ 1.9.0 | Neural network and backpropagation |
| `numpy` | ≥ 1.21.0 | State vector processing |
| `matplotlib` | ≥ 3.4.0 | Live training chart |

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
