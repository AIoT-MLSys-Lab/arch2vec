#!/usr/bin/env bash

#python search_methods/reinforce.py --dim 16 --seed $s --bs 16 --output_path saved_logs/rl  --saved_arch2vec --emb_path arch2vec-nasbench101.pt
for s in {1..500}
	do
		python search_methods/reinforce.py --dim 16 --seed $s --bs 16 --output_path saved_logs/rl  --saved_arch2vec --emb_path arch2vec-model-nasbench101.pt
	done
