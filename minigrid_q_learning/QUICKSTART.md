# Quick Start: Training on Warehouse Grid

Here are the quickest ways to get started:

## 1. Train Your First Warehouse Agent (5 minutes)

```bash
python train_q_learning.py --env WarehouseGridSmall-v0 --episodes 2000 --save-path results/warehouse_first_model.pkl
```

This trains a Q-learning agent on an 8×8 warehouse with shelves arranged in aisles.

## 2. Watch Your Agent Navigate

```bash
python evaluate_q_learning.py --env WarehouseGridSmall-v0 --model-path results/warehouse_first_model.pkl --episodes 3
```

The agent will navigate the warehouse and try to reach the goal!

## 3. Train Longer for Better Performance

```bash
python train_q_learning.py --env WarehouseGridSmall-v0 --episodes 5000 --save-path results/warehouse_trained.pkl
```

More episodes = smarter agent. Typical progression:
- 2,000 episodes: 40-50% success rate
- 5,000 episodes: 70-80% success rate
- 10,000 episodes: 85-95% success rate

## Warehouse Environment Details

The warehouse is procedurally generated with:
- **Vertical shelves** (walls) arranged in aisles
- **Random agent start position** (always in an aisle)
- **Random goal position** (always accessible)
- **Scalable sizes**: Small (8×8), Medium (12×12), Large (16×16)

## Key Differences vs Empty Grid

| Feature | Empty Grid | Warehouse |
|---------|-----------|-----------|
| Obstacles | None | Shelves arranged as walls |
| Difficulty | Very easy | Medium (with obstacles) |
| State space | Small | Medium to large |
| Real-world relevance | Low | High (warehouse-like) |

## Next Steps

1. **Start with Small warehouse** (this one learns fastest)
2. **Try Tiny warehouse** for faster debugging:
   ```bash
   python train_q_learning.py --env WarehouseGridTiny-v0 --episodes 2000 --save-path results/tiny_warehouse.pkl
   ```
3. **Compare against empty grid** to see difference:
   ```bash
   python train_q_learning.py --env MiniGrid-Empty-5x5-v0 --episodes 2000 --save-path results/empty_baseline.pkl
   ```
4. **Try Lava challenge** once the warehouse task is working:
   ```bash
   python train_q_learning.py --env MiniGrid-LavaGapS7-v0 --episodes 5000 --save-path results/q_table_lava.pkl
   ```
5. **Try Medium or Large** warehouses after mastering the small one
6. **Tune hyperparameters** if learning is too slow/fast
7. **Check the detailed guide**: `WAREHOUSE_GUIDE.md`

## Example Commands for Different Scenarios

### Aggressive Exploration (More thorough learning)
```bash
python train_q_learning.py --env WarehouseGridSmall-v0 \
  --episodes 5000 \
  --epsilon-decay 0.995 \
  --alpha 0.15
```

### Quick Proof-of-Concept
```bash
python train_q_learning.py --env WarehouseGridSmall-v0 --episodes 1000
```

### Challenge Yourself (Large warehouse)
```bash
python train_q_learning.py --env WarehouseGridLarge-v0 --episodes 15000
```

## Success Metrics

After training, the evaluation output shows:
- **success_rate**: % of episodes where agent reached goal (target: >80%)
- **mean_return**: Average reward per episode (target: >0.8)

Good luck! 🤖📦
