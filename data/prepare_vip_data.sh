#!/usr/bin/env bash
set -uxo pipefail

export VERL_HOME=${VERL_HOME:-"$(pwd)"}
export TRAIN_FILE=${TRAIN_FILE:-"${VERL_HOME}/data/vip-dapo-math-17k.parquet"}
export aime24=${TEST_FILE:-"${VERL_HOME}/data/vip-aime-2024.parquet"}
export aime25=${TEST_FILE:-"${VERL_HOME}/data/vip-aime-2025.parquet"}
export amc=${TEST_FILE:-"${VERL_HOME}/data/vip-amc.parquet"}
export OVERWRITE=${OVERWRITE:-0}

mkdir -p "${VERL_HOME}/data"

if [ ! -f "${TRAIN_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${TRAIN_FILE}" "https://huggingface.co/datasets/JunHill/Dedup-4shot-DAPO-Math-17k/resolve/main/data/train-00000-of-00001.parquet?download=true"
fi

if [ ! -f "${aime24}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${aime24}" "https://huggingface.co/datasets/JunHill/4shot-aime24/resolve/main/data/train-00000-of-00001.parquet?download=true"
fi

if [ ! -f "${aime25}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${aime25}" "https://huggingface.co/datasets/JunHill/4shot-aime25/resolve/main/data/train-00000-of-00001.parquet?download=true"
fi

if [ ! -f "${amc}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${amc}" "https://huggingface.co/datasets/JunHill/4shot-amc/resolve/main/data/train-00000-of-00001.parquet?download=true"
fi
