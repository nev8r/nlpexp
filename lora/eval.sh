#!/bin/bash

# ================================
# 1️⃣ 设置多个实验目录
# ================================
BASE_DIRS=(
    # "/root/nlpexp/outputs/lora_rank4_lr1.5e-4"
    # "/root/nlpexp/outputs/lora_rank8_lr1e-4"
    "/root/nlpexp/outputs/lora_rank16_lr8e-5"
    "/root/nlpexp/outputs/lora_rank32_lr5e-5"
)
# echo "sleep 600s"
# ================================
# 2️⃣ 要评估的 checkpoint 编号
# ================================
CHECKPOINT_NUMS="110 220 327"

# ================================
# 3️⃣ 结果保存根目录
# ================================
RESULT_ROOT="./eval_results"
mkdir -p "$RESULT_ROOT"

# ================================
# 4️⃣ 主循环：对每个实验目录进行评估
# ================================
for BASE_DIR in "${BASE_DIRS[@]}"; do
    EXP_NAME=$(basename "$BASE_DIR")                # 提取目录名作为实验名
    RESULT_DIR="${RESULT_ROOT}/${EXP_NAME}"         # 为每个实验单独创建结果目录
    mkdir -p "$RESULT_DIR"

    echo "========================================"
    echo "Start evaluating experiment: $EXP_NAME"
    echo "Model base dir: $BASE_DIR"
    echo "Results will be saved under: $RESULT_DIR"
    echo "========================================"

    for num in $CHECKPOINT_NUMS; do
        MODEL_PATH="${BASE_DIR}/checkpoint-${num}-merged"
        OUTPUT_FILE="${RESULT_DIR}/checkpoint-${num}-merged_acc.txt"

        echo "----------------------------------------"
        echo "Evaluating model: $MODEL_PATH"
        echo "Result will be saved to: $OUTPUT_FILE"

        python infer_vllm.py \
            --model_path "$MODEL_PATH" \
            --output_file "$OUTPUT_FILE"

        if [ $? -eq 0 ]; then
            echo "✅ Evaluation for checkpoint-$num completed successfully"
        else
            echo "❌ Error: Evaluation for checkpoint-$num failed"
        fi
    done
done

echo "========================================"
echo "All evaluations finished."
echo "Results are organized under: $RESULT_ROOT"
