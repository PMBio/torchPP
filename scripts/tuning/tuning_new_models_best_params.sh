#!/usr/bin/env bash
# ============================================================
# Best per-(dataset, model) architecture overrides for the new neural TPPs,
# consumed by generate_main_run_commands.sh.
#
# Selected by max validation LL (best_val_ll) from the tuning sweep
# (generate_tuning_commands_new_models.sh).
# Keys are <dataset>:<flavor>, where flavor is one of:
#   intensity_free_gru | intensity_free_transformer | nhp | attnhp
# (two IFTPP flavors share model/intensity_free; the encoder is set via
# arch.iftpp_encoder in the params string below.)
#
# Notes on what was tuned vs fixed:
#   * NHP (single-layer CT-LSTM): only n_embd swept {32,64}; n_layer/n_head N/A.
#   * IFTPP-gru: n_layer {2,4} x n_embd {32,64} swept; encoder gru; K=64.
#   * AttNHP / IFTPP-xfmr: per-dataset predetermined transformer arch
#     (BEST_NEURAL from tuning_easytpp_best_params.sh), not swept; IFTPP K=64.
# ============================================================

declare -gA BEST_MODEL_PARAMS

# --- taxi ---
BEST_MODEL_PARAMS["taxi:nhp"]="arch.n_embd=64"
BEST_MODEL_PARAMS["taxi:attnhp"]="arch.n_layer=2 arch.n_head=2 arch.n_embd=32 arch.attnhp_time_emb_size=16"
BEST_MODEL_PARAMS["taxi:intensity_free_gru"]="arch.iftpp_encoder=gru arch.n_layer=4 arch.n_embd=32 arch.num_mix_components=64"
BEST_MODEL_PARAMS["taxi:intensity_free_transformer"]="arch.iftpp_encoder=transformer arch.n_layer=2 arch.n_head=2 arch.n_embd=32 arch.num_mix_components=64"

# --- stackoverflow ---
BEST_MODEL_PARAMS["stackoverflow:nhp"]="arch.n_embd=32"
BEST_MODEL_PARAMS["stackoverflow:attnhp"]="arch.n_layer=4 arch.n_head=4 arch.n_embd=32 arch.attnhp_time_emb_size=16"
BEST_MODEL_PARAMS["stackoverflow:intensity_free_gru"]="arch.iftpp_encoder=gru arch.n_layer=2 arch.n_embd=32 arch.num_mix_components=64"
BEST_MODEL_PARAMS["stackoverflow:intensity_free_transformer"]="arch.iftpp_encoder=transformer arch.n_layer=4 arch.n_head=4 arch.n_embd=32 arch.num_mix_components=64"

# --- retweet ---
BEST_MODEL_PARAMS["retweet:nhp"]="arch.n_embd=64"
BEST_MODEL_PARAMS["retweet:attnhp"]="arch.n_layer=4 arch.n_head=2 arch.n_embd=64 arch.attnhp_time_emb_size=16"
BEST_MODEL_PARAMS["retweet:intensity_free_gru"]="arch.iftpp_encoder=gru arch.n_layer=4 arch.n_embd=32 arch.num_mix_components=64"
BEST_MODEL_PARAMS["retweet:intensity_free_transformer"]="arch.iftpp_encoder=transformer arch.n_layer=4 arch.n_head=2 arch.n_embd=64 arch.num_mix_components=64"

# --- amazon ---
BEST_MODEL_PARAMS["amazon:nhp"]="arch.n_embd=64"
BEST_MODEL_PARAMS["amazon:attnhp"]="arch.n_layer=2 arch.n_head=4 arch.n_embd=32 arch.attnhp_time_emb_size=16"
BEST_MODEL_PARAMS["amazon:intensity_free_gru"]="arch.iftpp_encoder=gru arch.n_layer=2 arch.n_embd=64 arch.num_mix_components=64"
BEST_MODEL_PARAMS["amazon:intensity_free_transformer"]="arch.iftpp_encoder=transformer arch.n_layer=2 arch.n_head=4 arch.n_embd=32 arch.num_mix_components=64"

# --- taobao ---
BEST_MODEL_PARAMS["taobao:nhp"]="arch.n_embd=64"
BEST_MODEL_PARAMS["taobao:attnhp"]="arch.n_layer=2 arch.n_head=2 arch.n_embd=32 arch.attnhp_time_emb_size=16"
BEST_MODEL_PARAMS["taobao:intensity_free_gru"]="arch.iftpp_encoder=gru arch.n_layer=2 arch.n_embd=64 arch.num_mix_components=64"
BEST_MODEL_PARAMS["taobao:intensity_free_transformer"]="arch.iftpp_encoder=transformer arch.n_layer=2 arch.n_head=2 arch.n_embd=32 arch.num_mix_components=64"

# ============================================================
# UKB / MIMIC.
# IFTPP + NHP only; AttNHP not tuned on ukb/mimic → intentionally absent (skipped).
# CI / batching / eval-batch / checkpoint args all come from the generator profile
# (e.g. bucket_numerical_small@64×2 for NHP) — do not duplicate them here.
# ============================================================

# --- ukb (ukb_cluster_full) ---
# NHP tuning (n_embd {32,64}) did not converge in the time budget; latent dim fixed
# to 128 and training capped at 20k steps.
BEST_MODEL_PARAMS["ukb_cluster_full:nhp"]="arch.n_embd=128 training.num_steps=20000"
BEST_MODEL_PARAMS["ukb_cluster_full:intensity_free_gru"]="arch.iftpp_encoder=gru arch.n_layer=2 arch.n_embd=32 arch.num_mix_components=64"
BEST_MODEL_PARAMS["ukb_cluster_full:intensity_free_transformer"]="arch.iftpp_encoder=transformer arch.n_layer=12 arch.n_head=8 arch.n_embd=128 arch.num_mix_components=64"

# --- mimic (mimic_iv_icu_interventions_top116) ---
BEST_MODEL_PARAMS["mimic_iv_icu_interventions_top116:nhp"]="arch.n_embd=64"
BEST_MODEL_PARAMS["mimic_iv_icu_interventions_top116:intensity_free_gru"]="arch.iftpp_encoder=gru arch.n_layer=2 arch.n_embd=64 arch.num_mix_components=64"
BEST_MODEL_PARAMS["mimic_iv_icu_interventions_top116:intensity_free_transformer"]="arch.iftpp_encoder=transformer arch.n_layer=4 arch.n_head=4 arch.n_embd=32 arch.num_mix_components=64"
