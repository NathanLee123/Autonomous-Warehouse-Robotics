# Warehouse Grid Environment Guide

This guide explains how to use the custom warehouse grid environments with the Q-learning agent.

## Available Environments

Warehouse and baseline environments available in this project:

- `WarehouseGridTiny-v0` — tiny 6x6 warehouse for fast testing
- `WarehouseGridSmall-v0` — small 8x8 warehouse with wider aisles
- `WarehouseGridMedium-v0` — medium 12x12 warehouse
- `WarehouseGridLarge-v0` — large 16x16 warehouse for more challenge
- `MiniGrid-Empty-5x5-v0` — baseline empty grid
- `MiniGrid-Empty-8x8-v0` — larger empty grid baseline
- `MiniGrid-LavaGapS7-v0` — lava corridor challenge

## Available Warehouse Environments

Three warehouse environments are available, varying in difficulty:

1. **WarehouseGridSmall-v0** (8x8)
   - Wider aisles (easier navigation)
   - Good for initial learning
   - Smaller state space

2. **WarehouseGridMedium-v0** (12x12)
   - Standard aisle width
   - Balanced difficulty
   - Medium state space

3. **WarehouseGridLarge-v0** (16x16)
   - Narrow aisles (challenging)
   - More complex environment
   - Larger state space (requires more training)

## Training on Warehouse Environments

### Train on Tiny Warehouse (Fastest to Run)
```bash
python train_q_learning.py --env WarehouseGridTiny-v0 --episodes 5000 --save-path results/tiny_warehouse_q_table.pkl
```

### Train on Small Warehouse (Recommended for Start)
```bash
python train_q_learning.py --env WarehouseGridSmall-v0 --episodes 5000 --save-path results/warehouse_small_q_table.pkl
```

### Train on Medium Warehouse
```bash
python train_q_learning.py --env WarehouseGridMedium-v0 --episodes 10000 --save-path results/warehouse_medium_q_table.pkl
```

### Train on Large Warehouse
```bash
python train_q_learning.py --env WarehouseGridLarge-v0 --episodes 15000 --save-path results/warehouse_large_q_table.pkl
```

### Adjust Hyperparameters for Warehouse
```bash
# Lower epsilon decay for more exploration
python train_q_learning.py --env WarehouseGridSmall-v0 \
  --episodes 5000 \
  --epsilon-decay 0.995 \
  --alpha 0.15 \
  --save-path results/warehouse_tuned_q_table.pkl
```

## Evaluating Trained Models on Warehouse

### View Agent Performance (5 episodes with visualization)
```bash
python evaluate_q_learning.py --env WarehouseGridSmall-v0 \
  --model-path results/warehouse_small_q_table.pkl \
  --episodes 5 \
  --render-mode human
```

### Run Agent Without Visualization
```bash
python evaluate_q_learning.py --env WarehouseGridSmall-v0 \
  --model-path results/warehouse_small_q_table.pkl \
  --episodes 10 \
  --render-mode rgb_array
```

## Comparing Original vs Warehouse Environments

### Train on Original Empty Grid (Baseline)
```bash
python train_q_learning.py --env MiniGrid-Empty-5x5-v0 --episodes 5000 --save-path results/empty_5x5_q_table.pkl
```

### Train on Empty 8x8 (Similar to Warehouse Small)
```bash
python train_q_learning.py --env MiniGrid-Empty-8x8-v0 --episodes 5000 --save-path results/empty_8x8_q_table.pkl
```

### Train on Lava Corridor Challenge
```bash
python train_q_learning.py --env MiniGrid-LavaGapS7-v0 --episodes 5000 --save-path results/q_table_lava.pkl
```

## Warehouse Environment Features

- **Procedural Generation**: Each episode generates a random warehouse layout
- **Random Agent Position**: Agent starts at a random valid location
- **Random Goal Position**: Goal location varies per episode
- **Shelf Obstacles**: Vertical walls arranged as shelves/aisles
- **Scalable**: Easy to customize aisle width and shelf height

## Tips for Training

1. **Start Small**: Begin with WarehouseGridSmall-v0 to understand the environment
2. **More Episodes**: Warehouse environments benefit from longer training (10k+ episodes)
3. **Adjust Decay**: Consider lower epsilon decay (0.995-0.998) for warehouse tasks
4. **Monitor Learning**: Use the printed statistics to see if learning is progressing
5. **Test Frequently**: Use evaluation script to see actual agent behavior

## Customizing Warehouse Environment

Edit `warehouse_env.py` to create custom configurations:

```python
# Create custom warehouse in warehouse_env.py:
class WarehouseGridCustom(WarehouseGridEnv):
    def __init__(self, render_mode: str | None = None):
        super().__init__(
            width=14,           # Grid width
            height=14,          # Grid height
            aisle_width=3,      # Walkable space between shelves
            shelf_height=3,     # Height of shelf obstacles
            render_mode=render_mode,
        )
```

## Expected Performance

With 5000 episodes of training on WarehouseGridSmall-v0:
- Expected success rate: 60-80%
- Expected mean reward: 0.6-0.8
- Average episode length: gradually decreasing

With 10000+ episodes:
- Expected success rate: 80-95%
- Expected mean reward: 0.8-0.95

## Troubleshooting

**Q: Agent not learning (low success rate after 5000 episodes)**
- Try lower epsilon decay (0.995)
- Increase learning rate (--alpha 0.15)
- Start with WarehouseGridSmall-v0
- Increase training episodes

**Q: Training very slow**
- Start with fewer episodes first to verify setup
- Use smaller environment (WarehouseGridSmall-v0)
- The state space grows with grid size

**Q: Agent keeps bumping into walls**
- This is expected initially - more training is needed
- Try with lower epsilon-end (0.01-0.02) to encourage more exploitation
