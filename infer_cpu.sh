#!/bin/bash
# CPU inference script for TDATR
# Set TDATR_CPU_MODE=1 to enable CPU mode

export TDATR_CPU_MODE=1
export HYDRA_FULL_ERROR=1

USE_NAIIVE=true
USE_OCR_PLUG=false

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
echo "root_dir= $ROOT_DIR"
cd "$ROOT_DIR"

TEST_FILE=${1:-"YOUR_TEST_IMAGE_PATH_LIST.json"}
minigpt4_v=TDATR
CFG_NAME=config_cpu
py_name=infer

CKPT_PATH=${2:-"model.pt"}
echo "CKPT: $CKPT_PATH"

python "$ROOT_DIR/$minigpt4_v/eval/${py_name}.py" \
    --config-dir "$ROOT_DIR/configs/" \
    --config-name "$CFG_NAME" \
    common.user_dir="$ROOT_DIR/$minigpt4_v" \
    common.npu=false \
    +model.rectification_rotate_flag=false \
    +model.rectification_textline_height_flag=false \
    task.use_ocr_plug=$USE_OCR_PLUG \
    model.use_naiive=$USE_NAIIVE \
    model.lora.apply_lora=false \
    +model.use_vit_encoder=false \
    +model.use_donut_encoder=true \
    +model.use_cfgi=true \
    model.ckpt="$CKPT_PATH" \
    generation.prompt_path="$TEST_FILE" \
    generation.no_repeat_ngram_size=15 \
    generation.min_len=1 \
    generation.max_len=4096 \
    generation.temperature=0.5 \
    task.seed=42
