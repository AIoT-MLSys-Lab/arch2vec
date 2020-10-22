#!/usr/bin/env bash
for i in {16,}
	do
		for s in {1..500}
			do
				python search_methods/reinforce_search_NB201_8x8.py --latent_dim $i --seed $s --bs 16 --MAX_BUDGET 500000 --baseline 0.4 --gamma 0.4 --saved_arch2vec \
				--dataset_name cifar100 --output_path saved_logs/rl  --model_path model-nasbench201.pt
			done
	done
