#!/usr/bin/env bash
python models/pretraining_darts.py --dim 16 --cfg 4 --bs 32 --epochs 10 --hidden_dim 128 --dim 16  --data data/data_darts_counter600000.json --name darts
