#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NPO Training Script for OmniSQL
 open-unlearning  NPO 
"""

import os
import sys
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType

#  wandb， None
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging will be disabled")

# 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from npo_trainer import NPOTrainer
from data_utils import QADataset, ForgetRetainDataset, NPODataCollator


def parse_args():
    parser = argparse.ArgumentParser(description="NPO Training for OmniSQL")
    
    # 
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the SFT model")
    parser.add_argument("--forget_data", type=str, required=True,
                        help="Path to forget dataset (JSON)")
    parser.add_argument("--retain_data", type=str, required=True,
                        help="Path to retain dataset (JSON)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints")
    
    # NPO 
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO temperature parameter")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Weight for retain loss")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Weight for forget loss")
    
    # LoRA 
    parser.add_argument("--use_lora", action="store_true",
                        help="Whether to use LoRA")
    parser.add_argument("--lora_r", type=int, default=32,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")
    
    # 
    parser.add_argument("--max_length", type=int, default=3200,
                        help="Maximum sequence length")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="Learning rate scheduler type")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Maximum number of checkpoints to keep")
    
    # Wandb 
    parser.add_argument("--report_to", type=str, default="wandb",
                        help="Report to wandb or none")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Wandb run name")
    parser.add_argument("--wandb_project", type=str, default="npo_omnisql",
                        help="Wandb project name")
    
    # 
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bfloat16")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Use float16")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Use gradient checkpointing")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--debug", action="store_true", default=True,
                        help="Enable debug output for token-level forget")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 
    set_seed(args.seed)
    
    print("=" * 80)
    print("NPO Training Configuration")
    print("=" * 80)
    print(f"Model Path: {args.model_path}")
    print(f"Forget Data: {args.forget_data}")
    print(f"Retain Data: {args.retain_data}")
    print(f"Output Dir: {args.output_dir}")
    print(f"NPO Parameters: beta={args.beta}, alpha={args.alpha}, gamma={args.gamma}")
    print(f"Use LoRA: {args.use_lora}")
    if args.use_lora:
        print(f"  LoRA Config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Token-level Forget: enabled (debug={args.debug})")
    print(f"Training: epochs={args.num_train_epochs}, batch_size={args.per_device_train_batch_size}, "
          f"grad_accum={args.gradient_accumulation_steps}")
    print(f"Learning Rate: {args.learning_rate}")
    print("=" * 80)
    
    # 1.  tokenizer
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    
    # 2. 
    print("\n[2/6] Loading datasets...")
    forget_dataset = QADataset(args.forget_data, tokenizer, args.max_length)
    retain_dataset = QADataset(args.retain_data, tokenizer, args.max_length)
    train_dataset = ForgetRetainDataset(forget_dataset, retain_dataset, anchor="forget")
    print(f"✅ Datasets loaded:")
    print(f"   - Forget: {len(forget_dataset)} samples")
    print(f"   - Retain: {len(retain_dataset)} samples")
    print(f"   - Train: {len(train_dataset)} pairs")
    
    # 3.  data collator
    print("\n[3/6] Creating data collator...")
    data_collator = NPODataCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        debug=args.debug
    )
    print("✅ Data collator created")
    print(f"   - Token-level forget: enabled")
    print(f"   - Supported modes: First (single token), Whole (multiple tokens)")
    
    # 4. 
    print("\n[4/6] Loading models...")
    print("Loading training model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    #  LoRA
    if args.use_lora:
        print(f"Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        #  gradient checkpointing， requires_grad
        if args.gradient_checkpointing:
            print("Enabling gradient checkpointing for LoRA...")
            model.enable_input_require_grads()
    
    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    print("✅ Models loaded")
    
    # 5. 
    print("\n[5/6] Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=args.report_to,
        run_name=args.run_name or f"npo_beta{args.beta}_alpha{args.alpha}_gamma{args.gamma}",
        remove_unused_columns=False,  # ：
        ddp_find_unused_parameters=False,
        seed=args.seed,
    )
    
    #  wandb，
    if args.report_to == "wandb":
        if not WANDB_AVAILABLE:
            print("Warning: wandb not available, switching to no logging")
            training_args.report_to = "none"
        else:
            try:
                wandb.init(
                    project=args.wandb_project,
                    name=training_args.run_name,
                    config=vars(args)
                )
            except Exception as e:
                print(f"Warning: wandb init failed ({e}), switching to no logging")
                training_args.report_to = "none"
    
    print("✅ Training arguments configured")
    
    # 6.  NPO Trainer
    print("\n[6/6] Creating NPO Trainer...")
    trainer = NPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=args.beta,
        alpha=args.alpha,
        gamma=args.gamma,
        debug=args.debug,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    print("✅ NPO Trainer created")
    
    # 7. 
    print("\n" + "=" * 80)
    print("Starting NPO Training...")
    print("=" * 80)
    trainer.train()
    
    # 8. 
    print("\n" + "=" * 80)
    print("Saving final model...")
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    #  LoRA，
    if args.use_lora:
        print("Merging and saving LoRA weights...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(os.path.join(args.output_dir, "merged_model"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "merged_model"))
    
    print("✅ Training completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
