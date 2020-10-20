#!/usr/bin/env bash

python plot_scripts/visgraph.py \
--data_type darts \
--data_path data/data_darts_counter600000.json \
--emb_path pretrained/dim-16/arch2vec-darts.pt \
--output_path graphvisualization
