#!/usr/bin/env bash

python plot_scripts/visgraph.py \
--data_type nasbench201 \
--data_path data/cifar10_valid_converged.json \
--emb_path pretrained/dim-16/cifar10_valid_converged-arch2vec.pt \
--supervised_emb_path pretrained/dim-16/supervised_dngo_embedding_cifar10_nasbench201.npy \
--output_path graphvisualization