#!/usr/bin/env bash
set -uxo pipefail

set -a        
source .env
set +a

TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/vip-dapo-math-17k.parquet"}
AIME24=${TEST_FILE:-"${DATA_DIR}/vip-aime-2024.parquet"}
AIME25=${TEST_FILE:-"${DATA_DIR}/vip-aime-2025.parquet"}
AMC=${TEST_FILE:-"${DATA_DIR}/vip-amc.parquet"}
OVERWRITE=${OVERWRITE:-0}

mkdir -p "${DATA_DIR}"

if [ ! -f "${TRAIN_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${TRAIN_FILE}" "https://huggingface.co/datasets/JunHill/Dedup-4shot-DAPO-Math-17k/resolve/main/data/train-00000-of-00001.parquet?download=true"
fi

if [ ! -f "${AIME24}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${AIME24}" "https://huggingface.co/datasets/JunHill/4shot-aime24/resolve/main/data/train-00000-of-00001.parquet?download=true"
fi

if [ ! -f "${AIME25}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${AIME25}" "https://huggingface.co/datasets/JunHill/4shot-aime25/resolve/main/data/train-00000-of-00001.parquet?download=true"
fi

if [ ! -f "${AMC}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${AMC}" "https://huggingface.co/datasets/JunHill/4shot-amc/resolve/main/data/train-00000-of-00001.parquet?download=true"
fi