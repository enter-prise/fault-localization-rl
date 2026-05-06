#!/bin/bash

# 最优训练配置
python main.py --mode train \
  --repo_path "data_storage/repos/astropy" \
  --data_path "data_storage/splits/train.parquet" \
  --split train \
  --use_dense \
  --dense_model_path "./dense_retriever_model" \
  --use_llm \
  --timesteps 300000 \
  --max_steps 30 \
  --top_k_retrieval 5 \
  --learning_rate 0.0003 \
  --ent_coef 0.08 \
  --batch_size 128 \
  --n_steps 4096 \
  --n_epochs 10 \
  --gamma 0.99 \
  --model_output "best_model_optimal" \
  --device cuda \
  --tensorboard_log "./logs_optimal"