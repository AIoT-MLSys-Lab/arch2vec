#!/usr/bin/env bash

for s in {1..500}
	do
		python search_methods/supervised_dngo.py --dim 16 --seed $s --init_size 16 --topk 5 --output_path saved_logs/bo
	done
