import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from swanlab.integration.transformers import SwanLabCallback
from dataclasses import dataclass
from utils import get_math_dataset
from peft import LoraConfig, get_peft_model
import swanlab

@dataclass
class Args:
    # model
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B"
    
    # train
    checkpoint_dir: str = "/root/nlpexp/outputs"
    learning_rate: float = 2e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    bf16: bool = True
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 1024
    epochs: int = 3
    save_steps: int = 10
    save_strategy: str = "steps"
    max_grad_norm: float = 1.0
    eval_strategy: str = "steps"
    eval_steps: int = 10
    save_limit: int = 30
    # lora
    use_lora: bool = False
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    
    # swanlab config
    project : str = "qwen2.5-1.5b-sft-lora"
    exp_name: str = "no_lora"

def train(args):
    training_args = SFTConfig(
        output_dir=f"{args.checkpoint_dir}/{args.exp_name}",
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
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        max_grad_norm=args.max_grad_norm,
        eval_strategy=args.eval_strategy,  
        eval_steps=args.eval_steps,
        save_total_limit=args.save_limit,
        metric_for_best_model="eval_loss",
        report_to=None,  
    )

    # 加载基础模型（移除未定义的cache_dir参数）
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
        device_map="auto",  
        trust_remote_code=True  # 加载Qwen等模型可能需要
    )
    # 将模型移至GPU
    model = model.to("cuda")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # 设置pad token

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
    # 初始化SFT训练器
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=get_math_dataset(sft=True),
        eval_dataset=get_math_dataset(sft=True, split="test"),
        callbacks=[swanlab_callback],
    )
    trainer.train()
    swanlab.finish()



if __name__ == "__main__":
    configs = [
        # no LoRA
        dict(use_lora=False, learning_rate=2e-5, exp_name="no_lora_2e-5"),

        # LoRA rank 4~32 不同学习率组合
        # dict(use_lora=True, lora_rank=4, learning_rate=1.5e-4, exp_name="lora_rank4_lr1.5e-4"),
        # dict(use_lora=True, lora_rank=8, learning_rate=1e-4, exp_name="lora_rank8_lr1e-4"),
        # dict(use_lora=True, lora_rank=16, learning_rate=8e-5, exp_name="lora_rank16_lr8e-5"),
        # dict(use_lora=True, lora_rank=32, learning_rate=5e-5, exp_name="lora_rank32_lr5e-5"),
    ]

    for cfg in configs:
        args = Args(**cfg)
        print(f"\n==============================")
        print(f"开始训练实验：{args.exp_name}")
        print(f"LoRA: {args.use_lora}, rank={args.lora_rank if args.use_lora else 'N/A'}, lr={args.learning_rate}")
        print("==============================\n")
        train(args)