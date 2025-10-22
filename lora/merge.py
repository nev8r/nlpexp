import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="merge lora and base")
    parser.add_argument(
        "--base_model_path", 
        default=None,
        type=str, 
        required=True, 
        help="基础模型的路径"
        )
    parser.add_argument(
        "--lora_path", 
        default=None,
        type=str, 
        required=True,
        help="LoRA适配器的路径")
    parser.add_argument(
        "--save_path", 
        default=None,
        type=str, 
        required=True, 
        help="合并后模型的保存路径")
    
    
    args = parser.parse_args()

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_model = PeftModel.from_pretrained(base_model, args.lora_path)
    merged_model = lora_model.merge_and_unload()

    merged_model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    print(f"合并完成！模型和tokenizer已保存到：{args.save_path}")

if __name__ == "__main__":
    main()