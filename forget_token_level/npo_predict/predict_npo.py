#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NPO
:
1. omnisql_npo: {"id": "xxx", "question": "...", "answer": "..."}
2. spider: {"instruction": "...", "input": "...", "output": "..."}
"""

import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_path):
    """tokenizer"""
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    print(f"✅ Model loaded successfully")
    return model, tokenizer


def load_data(data_path, data_format="omnisql"):
    """
    
    
    Args:
        data_path: 
        data_format:  ("omnisql"  "spider")
    
    Returns:
        List of data items
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples from {data_path}")
    
    # 
    normalized_data = []
    for item in data:
        if data_format == "omnisql":
            # omnisql_npo: {"id": "xxx", "question": "...", "answer": "..."}
            normalized_data.append({
                "id": item.get("id", "unknown"),
                "question": item["question"],
                "answer": item.get("answer", "")
            })
        elif data_format == "spider":
            # spider: {"instruction": "...", "input": "...", "output": "..."}
            # instructioninputquestion
            question = item["instruction"] + "\n" + item["input"]
            normalized_data.append({
                "id": item.get("id", "unknown"),
                "question": question,
                "answer": item.get("output", "")
            })
        else:
            raise ValueError(f"Unknown data format: {data_format}")
    
    return normalized_data


def generate_prompt(question, tokenizer):
    """
    prompt
    llama3 chat template，
    : <|begin_of_text|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>


    """
    # llama3prompt，system
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    return prompt


def predict_batch(model, tokenizer, prompts, max_new_tokens=512, temperature=0.1):
    """
    
    
    Args:
        model: 
        tokenizer: tokenizer
        prompts: prompt
        max_new_tokens: token
        temperature: 
    
    Returns:
        List of predictions
    """
    # Tokenize
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=3200
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,  # greedy decoding
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    predictions = []
    for i, output in enumerate(outputs):
        # 
        input_len = inputs["input_ids"][i].shape[0]
        generated_ids = output[input_len:]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predictions.append(prediction.strip())
    
    return predictions


def predict(model_path, data_path, output_path, data_format="omnisql", 
            batch_size=4, max_new_tokens=512):
    """
    
    
    Args:
        model_path: 
        data_path: 
        output_path: 
        data_format: 
        batch_size: batch
        max_new_tokens: token
    """
    # 1. 
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # 2. 
    data = load_data(data_path, data_format)
    
    # 3. 
    results = []
    print(f"\nGenerating predictions...")
    
    for i in tqdm(range(0, len(data), batch_size)):
        batch_data = data[i:i+batch_size]
        
        # prompts
        prompts = [generate_prompt(item["question"], tokenizer) 
                   for item in batch_data]
        
        # 
        predictions = predict_batch(
            model, tokenizer, prompts, 
            max_new_tokens=max_new_tokens
        )
        
        # 
        for item, pred in zip(batch_data, predictions):
            # prompt ()
            full_prompt = generate_prompt(item["question"], tokenizer)
            
            results.append({
                "prompt": full_prompt,
                "predict": pred,
                "label": item["answer"]
            })
    
    # 4. 
    print(f"\nSaving results to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"✅ Predictions saved to {output_path}")
    print(f"Total samples: {len(results)}")


def main():
    parser = argparse.ArgumentParser(description="NPO")
    
    # 
    parser.add_argument("--model_path", type=str, required=True,
                        help="NPO")
    parser.add_argument("--data_path", type=str, required=True,
                        help="")
    parser.add_argument("--output_path", type=str, required=True,
                        help=" (.jsonl)")
    
    # 
    parser.add_argument("--data_format", type=str, default="omnisql",
                        choices=["omnisql", "spider"],
                        help=": omnisql  spider")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="token")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("NPO")
    print("=" * 80)
    print(f": {args.model_path}")
    print(f": {args.data_path}")
    print(f": {args.output_path}")
    print(f": {args.data_format}")
    print(f"Batch: {args.batch_size}")
    print(f"token: {args.max_new_tokens}")
    print("=" * 80)
    
    predict(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output_path,
        data_format=args.data_format,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens
    )


if __name__ == "__main__":
    main()
