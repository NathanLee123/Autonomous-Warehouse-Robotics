# MiniGrid Q-Learning Example

This project gives you two runnable Python files:

- `train_q_learning.py` — trains a tabular Q-learning agent on `MiniGrid-Empty-5x5-v0`
- `evaluate_q_learning.py` — loads the saved Q-table and runs the policy

## Why this example works with tabular Q-learning

MiniGrid observations are normally richer than what a Q-table can handle efficiently. This example uses a compact state representation based on:

- agent x position
- agent y position
- agent direction

That is enough for the simple `Empty-5x5` environment.

## Setup

Create and activate a virtual environment:

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train

```bash
python train_q_learning.py
```

That saves `q_table.pkl` in the current directory.

You can also override defaults:

```bash
python train_q_learning.py --episodes 10000 --save-path models/q_table.pkl
```

## Evaluate / Watch

```bash
python evaluate_q_learning.py --model-path q_table.pkl
```

If you want a few more episodes:

```bash
python evaluate_q_learning.py --model-path q_table.pkl --episodes 10
```

## Notes

- This is a teaching example, not a high-performance RL baseline.
- Restricting actions to left / right / forward speeds up learning.
- For harder MiniGrid tasks, move from tabular Q-learning to DQN or PPO.
