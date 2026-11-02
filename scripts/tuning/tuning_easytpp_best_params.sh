#!/usr/bin/env bash
# Best per-dataset EasyTPP tuning results.
#
# Sourced by the tuning and run generators. Two layers:
#   1) Stage-1 winners reused by the staged tuning scripts.
#   2) Full per-(dataset, model_type) best parameter strings reused by the
#      run generators.

# ── Stage-1 reuse maps ───────────────────────────────────────
declare -gA BEST_BASELINE_K_MAP
declare -gA BEST_KERNEL_K_MAP
declare -gA BEST_KERNEL_TF_MAP

declare -gA BEST_NEURAL_NL_MAP
declare -gA BEST_NEURAL_NH_MAP
declare -gA BEST_NEURAL_NE_MAP

BEST_BASELINE_K_MAP[taxi]=10
BEST_BASELINE_K_MAP[stackoverflow]=10
BEST_BASELINE_K_MAP[retweet]=15
BEST_BASELINE_K_MAP[amazon]=5
BEST_BASELINE_K_MAP[taobao]=15

BEST_KERNEL_K_MAP[taxi]=8
BEST_KERNEL_K_MAP[stackoverflow]=8
BEST_KERNEL_K_MAP[retweet]=8
BEST_KERNEL_K_MAP[amazon]=8
BEST_KERNEL_K_MAP[taobao]=8

BEST_KERNEL_TF_MAP[taxi]=5
BEST_KERNEL_TF_MAP[stackoverflow]=20
BEST_KERNEL_TF_MAP[retweet]=5
BEST_KERNEL_TF_MAP[amazon]=5
BEST_KERNEL_TF_MAP[taobao]=20

BEST_NEURAL_NL_MAP[taxi]=2
BEST_NEURAL_NL_MAP[stackoverflow]=4
BEST_NEURAL_NL_MAP[retweet]=4
BEST_NEURAL_NL_MAP[amazon]=2
BEST_NEURAL_NL_MAP[taobao]=2

BEST_NEURAL_NH_MAP[taxi]=2
BEST_NEURAL_NH_MAP[stackoverflow]=4
BEST_NEURAL_NH_MAP[retweet]=2
BEST_NEURAL_NH_MAP[amazon]=4
BEST_NEURAL_NH_MAP[taobao]=2

BEST_NEURAL_NE_MAP[taxi]=32
BEST_NEURAL_NE_MAP[stackoverflow]=32
BEST_NEURAL_NE_MAP[retweet]=64
BEST_NEURAL_NE_MAP[amazon]=32
BEST_NEURAL_NE_MAP[taobao]=32

# Fill these in after stage-1 tuning for additional datasets.
# BEST_BASELINE_K_MAP[earthquake]=...
# BEST_KERNEL_K_MAP[earthquake]=...
# BEST_KERNEL_TF_MAP[earthquake]=...
# BEST_NEURAL_NL_MAP[earthquake]=...
# BEST_NEURAL_NH_MAP[earthquake]=...
# BEST_NEURAL_NE_MAP[earthquake]=...

# BEST_BASELINE_K_MAP[volcano]=...
# BEST_KERNEL_K_MAP[volcano]=...
# BEST_KERNEL_TF_MAP[volcano]=...
# BEST_NEURAL_NL_MAP[volcano]=...
# BEST_NEURAL_NH_MAP[volcano]=...
# BEST_NEURAL_NE_MAP[volcano]=...

# ── Full best params for replica runs ───────────────────────
# Key format: dataset:model_type
# Value format: exact extra train.py arguments beyond common config and LR.
declare -gA BEST_MODEL_PARAMS

# taxi
BEST_MODEL_PARAMS[taxi:poisson]=""
BEST_MODEL_PARAMS[taxi:spline_poisson]="arch.spline_K=10 arch.spline_T_factor=1.25"
BEST_MODEL_PARAMS[taxi:hawkes]=""
BEST_MODEL_PARAMS[taxi:const_spline_kernel]="arch.spline_kernel_K=8 arch.spline_kernel_T_factor=5 arch.spline_kernel_knot_power=4 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[taxi:neural_poisson]="training.weight_decay=0.0 arch.dropout=0.1 arch.n_layer=2 arch.n_head=2 arch.n_embd=32"
BEST_MODEL_PARAMS[taxi:spline_exp_hawkes]="arch.spline_K=15 arch.spline_T_factor=1.25 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[taxi:spline_spline_kernel]="arch.spline_K=15 arch.spline_T_factor=1.25 arch.spline_kernel_K=8 arch.spline_kernel_T_factor=5 arch.spline_kernel_knot_power=4 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[taxi:inhibitive_const_exp_hawkes]="arch.inhibitive_clamping_fn=softplus"
BEST_MODEL_PARAMS[taxi:inhibitive_spline_spline_kernel]="arch.spline_K=15 arch.spline_T_factor=1.25 arch.spline_kernel_K=8 arch.spline_kernel_T_factor=5 arch.spline_kernel_knot_power=4 arch.inhibitive_clamping_fn=softplus eval.eval_batch_size=128"
BEST_MODEL_PARAMS[taxi:neural_log_linear]="training.weight_decay=0.0 arch.dropout=0.1 arch.n_layer=4 arch.n_head=2 arch.n_embd=32"
BEST_MODEL_PARAMS[taxi:neural_spline]="training.weight_decay=0.0 arch.dropout=0.1 arch.n_layer=4 arch.n_head=2 arch.n_embd=32 arch.spline_kernel_K=8 arch.spline_kernel_T_factor=5 arch.spline_kernel_knot_power=4 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[taxi:neural_mlp]="training.weight_decay=0.001 arch.dropout=0.1 arch.n_layer=4 arch.n_head=4 arch.n_embd=32 arch.mlp_n_hidden=32 arch.mlp_n_layers=3 arch.mlp_time_embd_dim=32 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[taxi:neural_hawkes]="training.weight_decay=0 arch.dropout=0.1 arch.spline_K=10 arch.spline_T_factor=1.25 arch.n_layer=2 arch.n_head=2 arch.n_embd=32 arch.neural_kernel_emb_dim=32 arch.neural_kernel_time_embd_dim=32 arch.neural_kernel_n_hidden=32 arch.neural_kernel_n_mlp_layers=2 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[taxi:easytpp_thp]="training.weight_decay=0 arch.dropout=0.1 arch.n_layer=2 arch.n_head=2 arch.n_embd=32 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[taxi:easytpp_sahp]="training.weight_decay=0 arch.dropout=0.1 arch.n_layer=2 arch.n_head=2 arch.n_embd=32 eval.eval_batch_size=128"

# stackoverflow
BEST_MODEL_PARAMS[stackoverflow:poisson]=""
BEST_MODEL_PARAMS[stackoverflow:spline_poisson]="arch.spline_K=10 arch.spline_T_factor=1.25"
BEST_MODEL_PARAMS[stackoverflow:hawkes]=""
BEST_MODEL_PARAMS[stackoverflow:const_spline_kernel]="arch.spline_kernel_K=8 arch.spline_kernel_T_factor=20 arch.spline_kernel_knot_power=4 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[stackoverflow:neural_poisson]="training.weight_decay=0.0 arch.dropout=0.1 arch.n_layer=4 arch.n_head=4 arch.n_embd=32"
BEST_MODEL_PARAMS[stackoverflow:spline_exp_hawkes]="arch.spline_K=10 arch.spline_T_factor=1.25 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[stackoverflow:spline_spline_kernel]="arch.spline_K=10 arch.spline_T_factor=1.25 arch.spline_kernel_K=8 arch.spline_kernel_T_factor=20 arch.spline_kernel_knot_power=4 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[stackoverflow:inhibitive_const_exp_hawkes]="arch.inhibitive_clamping_fn=softplus"
BEST_MODEL_PARAMS[stackoverflow:inhibitive_spline_spline_kernel]="arch.spline_K=10 arch.spline_T_factor=1.25 arch.spline_kernel_K=8 arch.spline_kernel_T_factor=20 arch.spline_kernel_knot_power=4 arch.inhibitive_clamping_fn=softplus eval.eval_batch_size=128"
BEST_MODEL_PARAMS[stackoverflow:neural_log_linear]="training.weight_decay=0.0 arch.dropout=0.1 arch.n_layer=4 arch.n_head=4 arch.n_embd=32"
BEST_MODEL_PARAMS[stackoverflow:neural_spline]="training.weight_decay=0.0 arch.dropout=0.1 arch.n_layer=4 arch.n_head=4 arch.n_embd=32 arch.spline_kernel_K=8 arch.spline_kernel_T_factor=5 arch.spline_kernel_knot_power=4 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[stackoverflow:neural_mlp]="training.weight_decay=0 arch.dropout=0.1 arch.n_layer=4 arch.n_head=4 arch.n_embd=32 arch.mlp_n_hidden=32 arch.mlp_n_layers=2 arch.mlp_time_embd_dim=32 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[stackoverflow:neural_hawkes]="training.weight_decay=0 arch.dropout=0.1 arch.spline_K=10 arch.spline_T_factor=1.25 arch.n_layer=4 arch.n_head=4 arch.n_embd=32 arch.neural_kernel_emb_dim=32 arch.neural_kernel_time_embd_dim=32 arch.neural_kernel_n_hidden=32 arch.neural_kernel_n_mlp_layers=2 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[stackoverflow:easytpp_thp]="training.weight_decay=0 arch.dropout=0.1 arch.n_layer=2 arch.n_head=2 arch.n_embd=32 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[stackoverflow:easytpp_sahp]="training.weight_decay=0 arch.dropout=0.1 arch.n_layer=2 arch.n_head=2 arch.n_embd=32 eval.eval_batch_size=128"

# retweet
BEST_MODEL_PARAMS[retweet:poisson]=""
BEST_MODEL_PARAMS[retweet:spline_poisson]="arch.spline_K=15 arch.spline_T_factor=1.25"
BEST_MODEL_PARAMS[retweet:hawkes]=""
BEST_MODEL_PARAMS[retweet:const_spline_kernel]="arch.spline_kernel_K=8 arch.spline_kernel_T_factor=5 arch.spline_kernel_knot_power=4 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[retweet:neural_poisson]="training.weight_decay=0.0 arch.dropout=0.1 arch.n_layer=4 arch.n_head=2 arch.n_embd=64"
BEST_MODEL_PARAMS[retweet:spline_exp_hawkes]="arch.spline_K=15 arch.spline_T_factor=1.25 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[retweet:spline_spline_kernel]="arch.spline_K=15 arch.spline_T_factor=1.25 arch.spline_kernel_K=8 arch.spline_kernel_T_factor=5 arch.spline_kernel_knot_power=4 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[retweet:inhibitive_const_exp_hawkes]="arch.inhibitive_clamping_fn=softplus"
BEST_MODEL_PARAMS[retweet:inhibitive_spline_spline_kernel]="arch.spline_K=15 arch.spline_T_factor=1.25 arch.spline_kernel_K=8 arch.spline_kernel_T_factor=5 arch.spline_kernel_knot_power=4 arch.inhibitive_clamping_fn=softplus eval.eval_batch_size=128"
BEST_MODEL_PARAMS[retweet:neural_log_linear]="training.weight_decay=0.0 arch.dropout=0.1 arch.n_layer=4 arch.n_head=4 arch.n_embd=32"
BEST_MODEL_PARAMS[retweet:neural_spline]="training.weight_decay=0.0 arch.dropout=0.1 arch.n_layer=4 arch.n_head=4 arch.n_embd=32 arch.spline_kernel_K=8 arch.spline_kernel_T_factor=5 arch.spline_kernel_knot_power=4 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[retweet:neural_mlp]="training.weight_decay=0.0001 arch.dropout=0.1 arch.n_layer=4 arch.n_head=4 arch.n_embd=32 arch.mlp_n_hidden=16 arch.mlp_n_layers=2 arch.mlp_time_embd_dim=32 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[retweet:neural_hawkes]="training.weight_decay=0 arch.dropout=0.1 arch.spline_K=15 arch.spline_T_factor=1.25 arch.n_layer=4 arch.n_head=2 arch.n_embd=64 arch.neural_kernel_emb_dim=32 arch.neural_kernel_time_embd_dim=32 arch.neural_kernel_n_hidden=32 arch.neural_kernel_n_mlp_layers=2 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[retweet:easytpp_thp]="training.weight_decay=0 arch.dropout=0.1 arch.n_layer=2 arch.n_head=2 arch.n_embd=32 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[retweet:easytpp_sahp]="training.weight_decay=0 arch.dropout=0.1 arch.n_layer=2 arch.n_head=2 arch.n_embd=32 eval.eval_batch_size=128"

# amazon
BEST_MODEL_PARAMS[amazon:poisson]=""
BEST_MODEL_PARAMS[amazon:spline_poisson]="arch.spline_K=5 arch.spline_T_factor=1.25"
BEST_MODEL_PARAMS[amazon:hawkes]=""
BEST_MODEL_PARAMS[amazon:const_spline_kernel]="arch.spline_kernel_K=8 arch.spline_kernel_T_factor=5 arch.spline_kernel_knot_power=4 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[amazon:neural_poisson]="training.weight_decay=0.0 arch.dropout=0.1 arch.n_layer=2 arch.n_head=4 arch.n_embd=32"
BEST_MODEL_PARAMS[amazon:spline_exp_hawkes]="arch.spline_K=15 arch.spline_T_factor=1.25 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[amazon:spline_spline_kernel]="arch.spline_K=15 arch.spline_T_factor=1.25 arch.spline_kernel_K=8 arch.spline_kernel_T_factor=5 arch.spline_kernel_knot_power=4 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[amazon:inhibitive_const_exp_hawkes]="arch.inhibitive_clamping_fn=softplus"
BEST_MODEL_PARAMS[amazon:inhibitive_spline_spline_kernel]="arch.spline_K=15 arch.spline_T_factor=1.25 arch.spline_kernel_K=8 arch.spline_kernel_T_factor=5 arch.spline_kernel_knot_power=4 arch.inhibitive_clamping_fn=softplus eval.eval_batch_size=128"
BEST_MODEL_PARAMS[amazon:neural_log_linear]="training.weight_decay=0.0 arch.dropout=0.1 arch.n_layer=4 arch.n_head=2 arch.n_embd=32"
BEST_MODEL_PARAMS[amazon:neural_spline]="training.weight_decay=0.0 arch.dropout=0.1 arch.n_layer=4 arch.n_head=2 arch.n_embd=32 arch.spline_kernel_K=8 arch.spline_kernel_T_factor=20 arch.spline_kernel_knot_power=4 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[amazon:neural_mlp]="training.weight_decay=0.001 arch.dropout=0.1 arch.n_layer=4 arch.n_head=4 arch.n_embd=32 arch.mlp_n_hidden=32 arch.mlp_n_layers=1 arch.mlp_time_embd_dim=32 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[amazon:neural_hawkes]="training.weight_decay=0 arch.dropout=0.1 arch.spline_K=5 arch.spline_T_factor=1.25 arch.n_layer=2 arch.n_head=4 arch.n_embd=32 arch.neural_kernel_emb_dim=32 arch.neural_kernel_time_embd_dim=32 arch.neural_kernel_n_hidden=32 arch.neural_kernel_n_mlp_layers=2 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[amazon:easytpp_thp]="training.weight_decay=0 arch.dropout=0.1 arch.n_layer=2 arch.n_head=2 arch.n_embd=32 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[amazon:easytpp_sahp]="training.weight_decay=0 arch.dropout=0.1 arch.n_layer=2 arch.n_head=2 arch.n_embd=32 eval.eval_batch_size=128"

# taobao
BEST_MODEL_PARAMS[taobao:poisson]=""
BEST_MODEL_PARAMS[taobao:spline_poisson]="arch.spline_K=15 arch.spline_T_factor=1.25"
BEST_MODEL_PARAMS[taobao:hawkes]=""
BEST_MODEL_PARAMS[taobao:const_spline_kernel]="arch.spline_kernel_K=8 arch.spline_kernel_T_factor=20 arch.spline_kernel_knot_power=4 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[taobao:neural_poisson]="training.weight_decay=0 arch.dropout=0.1 arch.n_layer=2 arch.n_head=2 arch.n_embd=32"
BEST_MODEL_PARAMS[taobao:spline_exp_hawkes]="arch.spline_K=15 arch.spline_T_factor=1.25 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[taobao:spline_spline_kernel]="arch.spline_K=15 arch.spline_T_factor=1.25 arch.spline_kernel_K=8 arch.spline_kernel_T_factor=5 arch.spline_kernel_knot_power=4 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[taobao:inhibitive_const_exp_hawkes]="arch.inhibitive_clamping_fn=softplus"
BEST_MODEL_PARAMS[taobao:inhibitive_spline_spline_kernel]="arch.spline_K=15 arch.spline_T_factor=1.25 arch.spline_kernel_K=8 arch.spline_kernel_T_factor=5 arch.spline_kernel_knot_power=4 arch.inhibitive_clamping_fn=softplus eval.eval_batch_size=128"
BEST_MODEL_PARAMS[taobao:neural_log_linear]="training.weight_decay=0 arch.dropout=0.1 arch.n_layer=4 arch.n_head=4 arch.n_embd=64"
BEST_MODEL_PARAMS[taobao:neural_spline]="training.weight_decay=0 arch.dropout=0.1 arch.n_layer=4 arch.n_head=4 arch.n_embd=64 arch.spline_kernel_K=8 arch.spline_kernel_T_factor=20 arch.spline_kernel_knot_power=4 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[taobao:neural_hawkes]="training.weight_decay=0 arch.dropout=0.1 arch.spline_K=15 arch.spline_T_factor=1.25 arch.n_layer=4 arch.n_head=4 arch.n_embd=64 arch.neural_kernel_emb_dim=32 arch.neural_kernel_time_embd_dim=32 arch.neural_kernel_n_hidden=32 arch.neural_kernel_n_mlp_layers=2 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[taobao:neural_spline_hawkes]="training.weight_decay=0 arch.dropout=0.1 arch.spline_K=15 arch.spline_T_factor=1.25 arch.spline_kernel_K=8 arch.spline_kernel_T_factor=5 arch.spline_kernel_knot_power=4 arch.n_layer=4 arch.n_head=4 arch.n_embd=64 arch.neural_kernel_emb_dim=32 arch.neural_kernel_time_embd_dim=32 arch.neural_kernel_n_hidden=32 arch.neural_kernel_n_mlp_layers=1 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[taobao:neural_mlp]="training.weight_decay=0.0001 arch.dropout=0.1 arch.n_layer=4 arch.n_head=4 arch.n_embd=64 arch.mlp_n_hidden=32 arch.mlp_n_layers=3 arch.mlp_time_embd_dim=32 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[taobao:easytpp_thp]="training.weight_decay=0 arch.dropout=0.1 arch.n_layer=2 arch.n_head=2 arch.n_embd=32 eval.eval_batch_size=128"
BEST_MODEL_PARAMS[taobao:easytpp_sahp]="training.weight_decay=0 arch.dropout=0.1 arch.n_layer=2 arch.n_head=2 arch.n_embd=32 eval.eval_batch_size=128"