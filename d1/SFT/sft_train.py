# import torch
# import argparse
# from transformers import AutoTokenizer, AutoModel, TrainingArguments
# from datasets import load_dataset
# from torch.utils.data import DataLoader
# from peft import LoraConfig, get_peft_model, TaskType
# import os
# from sft_trainer import *
# import torch.distributed as dist
# import random
# import numpy as np


# def init_seed(seed):
#     random.seed(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True


# # Initialize argument parser
# def parse_args():
#     parser = argparse.ArgumentParser()

#     # Hyperparameters
#     parser.add_argument(
#         "--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct", help="Name of the pretrained model"
#     )
#     parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
#     parser.add_argument(
#         "--max_length", type=int, default=4096, help="Maximum sequence length for tokenization"
#     )
#     parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
#     parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer")
#     parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default="/data0/devaansh",
#         help="Directory to save model checkpoints and logs",
#     )
#     parser.add_argument("--job_name", type=str, default="llada-s1", help="Job Name")
#     parser.add_argument("--train_data", type=str, default="simplescaling/s1K", help="Path to training data")
#     parser.add_argument(
#         "--debugging", action="store_true", help="Use while debugging model - only disables wandb logging"
#     )

#     return parser.parse_args()


# # Model loading with LoRA integration
# def load_model_and_tokenizer(args):
#     # Load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(
#         args.model_name, padding_side="right", trust_remote_code=True, use_fast=True
#     )

#     # Load model
#     model = AutoModel.from_pretrained(
#         args.model_name,
#         trust_remote_code=True,
#         torch_dtype=torch.bfloat16,
#     )

#     # LoRA configuration
#     lora_config = LoraConfig(
#         r=128,
#         lora_alpha=256,
#         target_modules=["q_proj", "k_proj", "v_proj"],
#         lora_dropout=0.05,
#         bias="none",
#         task_type=TaskType.CAUSAL_LM,
#     )

#     # Applying LoRA model
#     model = get_peft_model(model, lora_config)
#     model = model.to(torch.bfloat16)  # Cast fp32 lora params to bf16

#     return tokenizer, model


# # Dataset loading
# def load_data(args, tokenizer):
#     data = load_dataset(args.train_data, split="train")
#     train_data, eval_data = preprocess_dataset(data, tokenizer, args.max_length)
#     print("Train data length: ", len(train_data))
#     print("Eval data length: ", len(eval_data))
#     train_dataset = dLLMSFTDataset(train_data, tokenizer, args.max_length)
#     eval_dataset = dLLMSFTDataset(eval_data, tokenizer, args.max_length, eval=True)
#     return train_dataset, eval_dataset


# # Training setup
# def train_model(args, tokenizer, model):
#     # Load dataset
#     train_dataset, eval_dataset = load_data(args, tokenizer)

#     # Training arguments setup
#     training_args = TrainingArguments(
#         output_dir=os.path.join(args.output_dir, args.job_name),
#         num_train_epochs=args.num_epochs,
#         per_device_train_batch_size=args.batch_size,
#         gradient_accumulation_steps=args.grad_accum_steps,
#         evaluation_strategy="steps",
#         eval_steps=100,
#         logging_steps=2,
#         save_steps=100,
#         save_total_limit=20,
#         learning_rate=args.learning_rate,
#         load_best_model_at_end=True,
#         weight_decay=0.1,
#         max_grad_norm=1.0,
#         bf16=True,
#         report_to="wandb" if not args.debugging else "none",
#         remove_unused_columns=False,
#     )

#     # Create optimizer and scheduler
#     num_train_steps = int(
#         len(train_dataset)
#         * args.num_epochs
#         / (args.batch_size * args.grad_accum_steps * torch.cuda.device_count())
#     )
#     # Initialize Trainer with custom dLLMTrainer
#     trainer = dLLMTrainer(
#         model=model,
#         args=training_args,
#         data_collator=dLLMDataCollator(tokenizer=tokenizer, mask_token_id=126336, max_length=args.max_length),
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#     )

#     # Start training
#     trainer.train()


# if __name__ == "__main__":
#     init_seed(42)
#     # Parse command-line arguments
#     args = parse_args()

#     # Load model and tokenizer
#     tokenizer, model = load_model_and_tokenizer(args)

#     # Train the model
#     train_model(args, tokenizer, model)

import os
import argparse
import random
import numpy as np
import torch
import torch.distributed as dist

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig, get_peft_model, TaskType

from sft_trainer import (
    dLLMTrainer,
    dLLMDataCollator,
    dLLMSFTDataset,
    preprocess_dataset,
)


# -----------------------------
# Reproducibility & speed tweaks
# -----------------------------
def init_seed(seed: int = 42):
    set_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # allow TF32 on Ampere+ (A100) for speed in matmul/conv
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# -----------------------------
# CLI args
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()

    # Model / data
    p.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                   help="HF model id or path")
    p.add_argument("--train_data", type=str, default="simplescaling/s1K",
                   help="HF dataset id or local path understood by preprocess_dataset")
    p.add_argument("--max_length", type=int, default=4096,
                   help="Max sequence length for tokenization")
    p.add_argument("--mask_token_id", type=int, default=None,
                   help="Override mask token id (defaults to tokenizer.mask_token_id or 126336)")

    # Optim / schedule
    p.add_argument("--num_epochs", type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)

    # Runtime
    p.add_argument("--output_dir", type=str, default="/scratch/USER/checkpoints",
                   help="Where to write checkpoints/logs")
    p.add_argument("--job_name", type=str, default="llada-s1")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers (TrainingArguments.dataloader_num_workers)")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable gradient checkpointing to save VRAM")
    p.add_argument("--bf16", action="store_true",
                   help="Force bf16 precision inside Trainer (Accelerate may already set this)")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Checkpoint path to resume from (Trainer.resume_from_checkpoint)")

    # Logging / eval / save cadence
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=4)
    p.add_argument("--evaluation_strategy", type=str, default="steps",
                   choices=["no", "steps", "epoch"])
    p.add_argument("--debugging", action="store_true",
                   help="Disable W&B logging")

    return p.parse_args()


# -----------------------------
# Model & tokenizer
# -----------------------------
def load_model_and_tokenizer(args):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="right",
        trust_remote_code=True,
        use_fast=True,
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        # LLaDA often uses eos as pad
        tokenizer.pad_token = tokenizer.eos_token
        try:
            tokenizer.save_pretrained("./_tmp_tokenizer_with_pad")
        except Exception:
            pass

    # Mask token id
    mask_id = args.mask_token_id
    if mask_id is None:
        mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        # default used by many diffusion LMs (LLaDA mask id)
        mask_id = 126336
    args.mask_token_id = mask_id

    # Model (bidirectional transformer; trust_remote_code=True is required)
    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Optional VRAM saver
    if args.gradient_checkpointing:
        # disable cache for gradient checkpointing compatibility
        if hasattr(model, "config"):
            setattr(model.config, "use_cache", False)
        model.gradient_checkpointing_enable()

    # LoRA config (keep as in your original)
    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,  # keep (works with PEFT even though LLaDA is non-causal)
    )
    model = get_peft_model(model, lora_config)
    model = model.to(torch.bfloat16)

    return tokenizer, model


# -----------------------------
# Data
# -----------------------------
def load_data(args, tokenizer):
    # HF dataset id or local—preprocess_dataset handles structure
    data = load_dataset(args.train_data, split="train")
    train_data, eval_data = preprocess_dataset(data, tokenizer, args.max_length)
    print(f"Train data length: {len(train_data)}")
    print(f"Eval  data length: {len(eval_data)}")
    train_dataset = dLLMSFTDataset(train_data, tokenizer, args.max_length)
    eval_dataset  = dLLMSFTDataset(eval_data,  tokenizer, args.max_length, eval=True)
    return train_dataset, eval_dataset


# -----------------------------
# Train
# -----------------------------
def train_model(args, tokenizer, model):
    train_dataset, eval_dataset = load_data(args, tokenizer)

    effective_out = os.path.join(args.output_dir, args.job_name)
    os.makedirs(effective_out, exist_ok=True)

    # Pick report destination
    report_to = "none" if args.debugging else "wandb"

    training_args = TrainingArguments(
        output_dir=effective_out,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,

        # eval/log/save cadence
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps if args.evaluation_strategy == "steps" else None,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        # precision & columns
        bf16=args.bf16 or True,  # keep bf16 on A100; Accelerate may set this too
        remove_unused_columns=False,

        # misc
        dataloader_num_workers=args.num_workers,
        load_best_model_at_end=True,
        report_to=report_to,
    )

    trainer = dLLMTrainer(
        model=model,
        args=training_args,
        data_collator=dLLMDataCollator(
            tokenizer=tokenizer,
            mask_token_id=args.mask_token_id,
            max_length=args.max_length,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Resume if requested
    resume_ckpt = args.resume_from if args.resume_from else False

    trainer.train(resume_from_checkpoint=resume_ckpt)

    # Always save a final full checkpoint + tokenizer
    try:
        trainer.save_model(effective_out)
        tokenizer.save_pretrained(effective_out)
    except Exception as e:
        print(f"[WARN] final save failed: {e}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    init_seed(42)
    args = parse_args()

    # Expand "~/…" for output dir early
    args.output_dir = os.path.expanduser(args.output_dir)

    tokenizer, model = load_model_and_tokenizer(args)
    train_model(args, tokenizer, model)
