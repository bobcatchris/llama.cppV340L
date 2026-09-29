#!/usr/bin/env bash
export HOME=/home/chris
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0=*
export BOOT_SCRIPT=/home/chris/launch_tp3_200k_therock.sh
cd /media/chris/ssd128/llamacpp/llama.cpp
exec docs/amd-port/tests/run_tp3_guards.sh --no-boot
