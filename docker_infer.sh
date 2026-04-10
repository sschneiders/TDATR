#!/bin/bash
set -e
# Run TDATR CPU inference in Docker
# Usage: ./docker_infer.sh <image_list.json> [output_dir]

IMAGE_LIST=${1:-"test_images.json"}
OUTPUT_DIR=${2:-"./output"}

sudo docker build -t tdatr-cpu .

sudo docker run --rm \
    -v "$(pwd)/$IMAGE_LIST:/app/test_images.json:ro" \
    -v "$(pwd)/$OUTPUT_DIR:/app/output" \
    tdatr-cpu \
    --config-dir configs/ \
    --config-name config_cpu \
    common.user_dir=TDATR \
    common.npu=false \
    +model.rectification_rotate_flag=false \
    +model.rectification_textline_height_flag=false \
    task.use_ocr_plug=false \
    model.use_naiive=true \
    model.lora.apply_lora=false \
    +model.use_vit_encoder=false \
    +model.use_donut_encoder=true \
    +model.use_cfgi=true \
    model.ckpt=model.pt \
    generation.prompt_path=test_images.json \
    generation.no_repeat_ngram_size=15 \
    generation.min_len=1 \
    generation.max_len=4096 \
    generation.temperature=0.5 \
    task.seed=42
