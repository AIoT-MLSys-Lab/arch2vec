#!/usr/bin/env bash
python search_methods/reinforce_darts.py --max_budgets 100 --inner_epochs 50 --objective 0.95 --train_portion 0.9 --dim 16 --seed 3 --bs 16 --output_path saved_logs/rl  --saved_arch2vec --logging_path darts-rl

