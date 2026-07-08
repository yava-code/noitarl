

https://github.com/user-attachments/assets/b5d99449-3ead-447d-965a-2d14676631f5




# 🧙‍♂️ Noita RL — Reinforcement Learning Agent for Noita

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build](https://img.shields.io/github/actions/workflow/status/yava-code/noitarl/tests.yml?label=tests)](.github/workflows/tests.yml)

Train an AI agent to play **Noita** — the physics-based roguelike — using reinforcement learning (PPO).

> 🎯 **Goal**: Teach an agent to navigate, fight, and survive in Noita's procedurally generated world using only pixel observations and game state data.

---

## ✨ Features

- **PPO Training** — Proximal Policy Optimization with custom reward shaping
- **Lua-Python Bridge** — Real-time communication between Noita's Lua mod and Python training loop via `pollnet.dll`
- **Custom Environment** — Gym-compatible wrapper with pixel observations, health, mana, and inventory state
- **Episode Tracking** — Automatic logging of rewards, deaths, and progression metrics
- **CI/CD** — Automated tests on every push

---

## 🏗️ Architecture

```
┌─────────────────┐         pollnet.dll         ┌──────────────────┐
│   Noita Game    │ ◄─────────────────────────► │  Python Trainer  │
│   (Lua Mod)     │        TCP / Shared          │  (PPO Agent)     │
│                 │         Memory               │                  │
└─────────────────┘                              └──────────────────┘
       │                                                  │
       ▼                                                  ▼
  Game State                                         Neural Network
  - Pixel buffer                                     - CNN encoder
  - HP / Mana                                        - Policy head
  - Inventory                                        - Value head
  - Position                                         - Reward shaping
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Noita** (Steam) — must be installed and launched at least once
- **Windows** (for `pollnet.dll` integration)

### Installation

```bash
# Clone the repository
git clone https://github.com/yava-code/noitarl.git
cd noitarl

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your Noita path and training settings
```

### Training

```bash
# Start training
python train.py

# Evaluate a trained model
python eval.py --checkpoint checkpoints/latest.pt
```

### Configuration

Key settings in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NOITA_PATH` | Auto-detect | Path to Noita installation |
| `LEARNING_RATE` | 3e-4 | PPO learning rate |
| `GAMMA` | 0.99 | Discount factor |
| `BATCH_SIZE` | 256 | Training batch size |
| `MAX_STEPS` | 10_000_000 | Total training steps |

---

## 📊 Training Progress

| Metric | Value |
|--------|-------|
| Episodes logged | See `data/episode_history.csv` |
| Best reward | Tracked in training logs |
| Checkpoints | Saved to `checkpoints/` |

---

## 📁 Project Structure

```
noitarl/
├── train.py              # Main training script
├── eval.py               # Evaluation script
├── config.py             # Configuration parameters
├── callbacks.py          # Training callbacks & logging
├── init.lua              # Noita Lua mod (game-side)
├── bin/
│   └── pollnet.dll       # Lua-Python communication library
├── data/
│   ├── episode_history.csv   # Episode metrics
│   └── schemas/              # Game state XML schemas
├── docs/
│   ├── lua_api_documentation.txt
│   ├── component_documentation.txt
│   └── Noita-ModdingAgreement-v100.rtf
├── .github/workflows/
│   └── tests.yml         # CI pipeline
└── ROADMAP.md            # Development roadmap
```

---

## 🗺️ Roadmap

See [ROADMAP.md](ROADMAP.md) for the full development plan.

**Current priorities:**
- [ ] Improve reward shaping for exploration
- [ ] Add multi-objective training (survival + progression)
- [ ] Implement curriculum learning
- [ ] Add wandb integration for experiment tracking

---

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Noita](https://noitagame.com/) by Nolla Games — incredible game with amazing modding support
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) — RL framework
- [pollnet](https://github.com/ikarth/pollnet) — Lua-Python communication library

---

<p align="center">Made with ❤️ and a lot of dead agents</p>
