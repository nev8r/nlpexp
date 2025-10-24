import os

# 输入文件（每行英文和中文用 tab 或固定空格分隔）
input_file = "/root/nlpexp/translation/data/raw_text.txt"

# 输出文件
output_en = "./data/train.en"
output_zh = "./data/train.zh"

en_lines = []
zh_lines = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # 先尝试 tab 分隔
        if "\t" in line:
            en, zh = line.split("\t", 1)
        # 如果没有 tab，尝试 4 个空格分隔
        elif "    " in line:
            en, zh = line.split("    ", 1)
        else:
            # 如果无法分隔，跳过或报错
            print(f"跳过无法解析的行: {line}")
            continue

        en_lines.append(en.strip())
        zh_lines.append(zh.strip())

# 保存英文和中文
with open(output_en, "w", encoding="utf-8") as f:
    f.write("\n".join(en_lines))

with open(output_zh, "w", encoding="utf-8") as f:
    f.write("\n".join(zh_lines))

print(f"处理完成: {len(en_lines)} 对句子")
print(f"英文文件: {os.path.abspath(output_en)}")
print(f"中文文件: {os.path.abspath(output_zh)}")
