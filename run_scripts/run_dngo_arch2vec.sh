#!/usr/bin/env bash
for s in {1..500}
	do
	  python search_methods/dngo.py --dim 16 --seed $s --output_path saved_logs/bo --emb_path arch2vec-model-nasbench101.pt --init_size 16 --topk 5
	done
