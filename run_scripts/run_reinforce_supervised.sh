#!/usr/bin/env bash

for s in {1..500}
	do
		python search_methods/supervised_reinforce.py --dim 16 --seed $s --bs 16 --output_path saved_logs/rl
	done
