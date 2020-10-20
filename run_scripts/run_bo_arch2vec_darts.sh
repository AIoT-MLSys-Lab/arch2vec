#!/usr/bin/env bash
python search_methods/dngo_darts.py --max_budgets 100 --inner_epochs 50 --objective 0.95 --train_portion 0.9 --dim 16 --seed 3 --output_path saved_logs/bo --init_size 16 --batch_size 5 --logging_path darts-bo
