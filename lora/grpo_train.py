import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from swanlab.integration.transformers import SwanLabCallback
from utils import get_math_dataset
from reward import REWARD_FUNCS
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass
import swanlab

@dataclass
class Args:
    # model
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B"
    
    # train
    checkpoint_dir: str = "/root/nlpexp/outputs"
    learning_rate: float = 1.5e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    bf16: bool = True
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_generations: int = 4
    max_prompt_length: int = 512
    max_completion_length: int = 1024
    epochs: int = 3
    save_steps: int = 10
    save_strategy: str = "steps"
    max_grad_norm: float = 1.0
    eval_strategy: str = "steps"
    eval_steps: int = 10
    save_limit: int = 30
    # vllm
    use_vllm: bool = False
    vllm_device: str = "cuda"
    vllm_gpu_ratio: float = 0.9
    # reward
    reward_funcs: str = "math_rewards_func"
    # lora
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    
    # swanlab config
    project: str = "qwen2.5-grpo-lora"
    exp_name: str = "grpo_lora_rank32"

def train(args):
    model_name = args.model_name_or_path.split('/')[-1]
    output_dir = f"{args.checkpoint_dir}/{model_name}_grpo_loraexp_rank_{args.lora_rank}" if args.use_lora else f"{args.checkpoint_dir}/{model_name}_grpo_no_lora"
    
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.epochs,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        max_grad_norm=args.max_grad_norm,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_limit,
        metric_for_best_model="eval_loss",
        use_vllm=args.use_vllm,
        vllm_device=args.vllm_device,
        vllm_gpu_memory_utilization=args.vllm_gpu_ratio,
        report_to=None
    )

    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
        device_map="auto",
        trust_remote_code=True
    )
    # 将模型移至GPU
    model = model.to("cuda")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # 设置pad token

    reward_funcs = [REWARD_FUNCS[func.strip()] for func in args.reward_funcs.split(',')]

    swanlab.init(
        project=args.project, 
        experiment_name=args.exp_name  
    )
    
    # 初始化swanlab回调
    swanlab_callback = SwanLabCallback(
        project=args.project, 
        experiment_name=args.exp_name 
    )

    if args.use_lora:
        # 配置LoRA参数
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        # 应用LoRA到模型
        model = get_peft_model(model, lora_config)
        print("LoRA模块已添加到模型中。")

    
    train_dataset = get_math_dataset()
    eval_dataset = get_math_dataset(split = "test")
    
    # 初始化GRPO训练器
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[swanlab_callback],
    )
    trainer.train()

if __name__ == "__main__":
    args = Args()
    args.model_name_or_path = "Qwen/Qwen2.5-1.5B"
    train(args)