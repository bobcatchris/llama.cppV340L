#!/usr/bin/env bash
# DP-3 A1 arm: tree AR enabled (root-owned setsid; survives session cleanup)
export HOME=/home/chris
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0=*
export NCCL_P2P_DISABLE=1
export GGML_CUDA_AR_TREE=1
cd /media/chris/ssd128/llamacpp/llama.cpp
exec docs/amd-port/tests/run_tp3_guards.sh
