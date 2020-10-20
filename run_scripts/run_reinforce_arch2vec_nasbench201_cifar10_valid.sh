#!/usr/bin/env bash
for i in {16,}
	do
		for s in {1..500}
			do
				python search_methods/reinforce_search_NB201_8x8.py --latent_dim $i --seed $s --bs 16 --gamma 0.4 --baseline 0.4 \
				--output_path saved_logs/rl  --saved_arch2vec \
				--dataset_name cifar10_valid_converged --MAX_BUDGET 12000 --model_path model-vae-nasbench201-seed3-epoch6.pt
			done
	done
