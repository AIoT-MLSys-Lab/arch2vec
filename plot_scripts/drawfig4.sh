#!/usr/bin/env bash

python plot_scripts/visdensity.py \
--emb_path pretrained/dim-16/arch2vec-model-nasbench101.pt \
--supervised_emb_path pretrained/dim-16/supervised_dngo_embedding_nasbench101.npy \
--output_path density/nas101