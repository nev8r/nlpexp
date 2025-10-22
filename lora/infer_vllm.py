import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import argparse  
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams
from reward import strict_reward_func
from utils import get_math_dataset


def test_vllm(model_path, output_file):
    # 初始化模型和tokenizer
    llm = LLM(model=model_path, gpu_memory_utilization=0.85)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def infer_vllm_batch(prompts):
        # 批量应用chat template
        texts = []
        for prompt in prompts:
            text = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)
        
        # 采样参数
        sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            top_k=50,
            repetition_penalty=1.0,
            max_tokens=2048
        )
        
        # 批量生成
        outputs = llm.generate(texts, sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]
        return generated_texts

    # 加载数据集
    dataset = get_math_dataset(split='test')
    print(f"Dataset size: {len(dataset)}")
    prompts = [item['prompt'] for item in dataset]

    # 批量推理
    print(f"Starting inference for model: {model_path}...")
    all_llm_answers = infer_vllm_batch(prompts)
    print("Inference completed. Calculating rewards...")

    # 计算准确率
    true_num = 0
    for i in tqdm(range(len(dataset))):
        llm_answer = all_llm_answers[i]
        reward = strict_reward_func(llm_answer, [dataset[i]["answer"]])
        true_num += reward[0] / 2.0  # 保持原逻辑
    
    acc = true_num / len(dataset)
    print(f"Model: {model_path}, Accuracy: {acc}")

    # 将结果写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"model_path: {model_path}\n")
        f.write(f"dataset_size: {len(dataset)}\n")
        f.write(f"accuracy: {acc:.6f}\n")  # 保留6位小数
    print(f"Result saved to: {output_file}")


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Evaluate merged model accuracy")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the merged model")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save accuracy result")
    args = parser.parse_args()

    # 调用评估函数
    test_vllm(args.model_path, args.output_file)

