#!/usr/bin/env bash
for i in {16,}
	do
		for s in {1..500}
			do
				python search_methods/dngo_search_NB201_8x8.py --dim $i --seed $s --output_path saved_logs/bo --init_size 16 --batch_size 1 \
				--dataset_name cifar100 --MAX_BUDGET 500000
			done
	done
