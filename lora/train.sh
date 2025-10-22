#!/bin/bash

# 确保日志目录存在，不存在则创建
mkdir -p logs

# 生成带时间戳的日志文件名（格式：年-月-日_时-分-秒）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/sft_train_${TIMESTAMP}.log"


python sft_train.py > "${LOG_FILE}" 2>&1 | ts &

echo "训练脚本已后台启动！"
echo "进程ID: $!"
echo "日志文件: ${LOG_FILE}"
echo "查看日志可执行: tail -f ${LOG_FILE}"