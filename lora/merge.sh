#!/bin/bash
# ================================
# 合并多个 LoRA 实验的 checkpoint
# ================================

# 基础模型路径（保持不变）
BASE_MODEL="Qwen/Qwen2.5-1.5B"

# 所有实验目录（不同 rank/lr）
EXPERIMENTS=(
    "lora_rank4_lr1.5e-4"
    "lora_rank8_lr1e-4"
    "lora_rank16_lr8e-5"
    "lora_rank32_lr5e-5"
)

# 要合并的 checkpoint 编号
CHECKPOINTS=(110 220 327)

# 输出根目录
ROOT_DIR="/root/nlpexp/outputs"

# ================================
# 双层循环执行 merge.py
# ================================
for exp in "${EXPERIMENTS[@]}"; do
  echo "========================================"
  echo "Processing experiment: $exp"
  echo "----------------------------------------"
  
  for num in "${CHECKPOINTS[@]}"; do
    LORA_PATH="${ROOT_DIR}/${exp}/checkpoint-${num}"
    SAVE_PATH="${ROOT_DIR}/${exp}/checkpoint-${num}-merged"

    echo "Merging checkpoint-${num} for ${exp}"
    echo "LoRA path:  ${LORA_PATH}"
    echo "Save path:  ${SAVE_PATH}"
    
    python merge.py \
      --base_model_path "$BASE_MODEL" \
      --lora_path "$LORA_PATH" \
      --save_path "$SAVE_PATH"

    # 检查执行是否成功
    if [ $? -eq 0 ]; then
      echo "✅ checkpoint-${num} merged successfully for ${exp}"
    else
      echo "❌ Error merging checkpoint-${num} for ${exp}"
    fi
    echo "----------------------------------------"
  done
done

echo "========================================"
echo "All merges completed!"
