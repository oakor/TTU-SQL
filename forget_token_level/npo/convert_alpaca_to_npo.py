#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import random
import argparse
from pathlib import Path


def load_exec_results(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_question_from_prompt(prompt):
    user_start = prompt.find('<|start_header_id|>user<|end_header_id|>')
    if user_start == -1:
        return prompt 
    
    user_start += len('<|start_header_id|>user<|end_header_id|>')
    user_end = prompt.find('<|eot_id|>', user_start)
    
    if user_end == -1:
        question = prompt[user_start:].strip()
    else:
        question = prompt[user_start:user_end].strip()
    
    return question


def split_by_exec_match(data, forget_size=10000, retain_size=10000, seed=42):
    random.seed(seed)

    false_samples = []
    true_samples = []
    
    for item in data:
        if item.get('exec_match') == False:
            false_samples.append(item)
        elif item.get('exec_match') == True:
            true_samples.append(item)
    
    print(f":")
    print(f"  - exec_match=false: {len(false_samples)} ")
    print(f"  - exec_match=true: {len(true_samples)} ")
    print(f"  - : {len(data)} ")
    
    random.shuffle(false_samples)
    random.shuffle(true_samples)
    

    forget_samples_from_false = false_samples[:forget_size]
    remaining_false = false_samples[forget_size:]
    
    forget_data = []
    for item in forget_samples_from_false:
        forget_data.append({
            'item': item,
            'answer_source': 'predict'  # 
        })
    
    print(f"\nForget set :")
    print(f"  -  false : {len(forget_data)}  ( predict)")
    
    # 2.  retain set:  true  retain_size ， predict
    retain_samples_from_true = true_samples[:retain_size]
    
    retain_data = []
    for item in retain_samples_from_true:
        retain_data.append({
            'item': item,
            'answer_source': 'predict'  # 
        })
    
    print(f"\nRetain set :")
    print(f"  -  true : {len(retain_data)}  ( predict)")
    
    # 3.  retain ， false ， label
    if len(retain_data) < retain_size:
        needed = retain_size - len(retain_data)
        print(f"  - Retain set  {retain_size} ， {needed} ")
        
        supplement_samples = remaining_false[:needed]
        for item in supplement_samples:
            retain_data.append({
                'item': item,
                'answer_source': 'label'  # 
            })
        
        print(f"  -  false : {len(supplement_samples)}  ( label)")
    
    print(f"\n:")
    print(f"  - Forget set: {len(forget_data)} ")
    print(f"  - Retain set: {len(retain_data)} ")
    
    return forget_data, retain_data


def convert_to_qa_format(split_data):
    """
     QA 
    
    : split_data = [{'item': {...}, 'answer_source': 'predict'/'label'}, ...]
    
     QA :
    {
        "id": "xxx",
        "question": "...",
        "answer": "...",
        "answer_source": "predict/label",
        "original_exec_match": true/false
    }
    """
    qa_data = []
    
    for entry in split_data:
        item = entry['item']
        answer_source = entry['answer_source']
        
        #  prompt 
        question = extract_question_from_prompt(item.get('prompt', ''))
        
        #  answer_source 
        if answer_source == 'predict':
            answer = item.get('predict', '')
        else:  # 'label'
            answer = item.get('label', '')
        
        qa_item = {
            "id": str(item.get('index', len(qa_data))),
            "question": question,
            "answer": answer,
            "answer_source": answer_source,
            "original_exec_match": item.get('exec_match', None),
            "original_error": item.get('error', None)
        }
        qa_data.append(qa_item)
    
    return qa_data


def save_json(data, file_path):
    """ JSON """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f" {len(data)} : {file_path}")


def main():
    parser = argparse.ArgumentParser(
        description=' NPO '
    )
    parser.add_argument(
        '--input', 
        type=str, 
        default='generated_predictions_with_exec_unified.json',
        help=''
    )
    parser.add_argument(
        '--output_dir', 
        type=str,
        default='../data/omnisql_npo',
        help=''
    )
    parser.add_argument(
        '--forget_size', 
        type=int, 
        default=10000,
        help=' ( 10000)'
    )
    parser.add_argument(
        '--retain_size', 
        type=int, 
        default=10000,
        help=' ( 10000)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help=''
    )
    
    args = parser.parse_args()
    
    # 
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(" NPO ")
    print("=" * 80)
    
    print(f"\n: {args.input}")
    data = load_exec_results(args.input)
    print(f": {len(data)}")
    
    #  exec_match  forget  retain
    print(f"\n{'='*80}")
    print(" Forget  Retain ")
    print(f"{'='*80}")
    print(f"Forget : {args.forget_size}")
    print(f"Retain : {args.retain_size}")
    print()
    
    forget_data, retain_data = split_by_exec_match(
        data, 
        forget_size=args.forget_size,
        retain_size=args.retain_size,
        seed=args.seed
    )
    
    #  QA 
    print(f"\n{'='*80}")
    print(" QA ")
    print(f"{'='*80}")
    forget_qa = convert_to_qa_format(forget_data)
    retain_qa = convert_to_qa_format(retain_data)
    
    # 
    forget_predict_count = sum(1 for x in forget_qa if x['answer_source'] == 'predict')
    forget_label_count = sum(1 for x in forget_qa if x['answer_source'] == 'label')
    retain_predict_count = sum(1 for x in retain_qa if x['answer_source'] == 'predict')
    retain_label_count = sum(1 for x in retain_qa if x['answer_source'] == 'label')
    
    print(f"\nForget set :")
    print(f"  -  predict: {forget_predict_count} ")
    print(f"  -  label: {forget_label_count} ")
    
    print(f"\nRetain set :")
    print(f"  -  predict: {retain_predict_count} ")
    print(f"  -  label: {retain_label_count} ")
    
    # 
    print(f"\n{'='*80}")
    print("")
    print(f"{'='*80}")
    forget_file = output_dir / "forget.json"
    retain_file = output_dir / "retain.json"
    
    save_json(forget_qa, forget_file)
    save_json(retain_qa, retain_file)
    
    # 
    stats = {
        "total_samples": len(data),
        "forget_samples": len(forget_qa),
        "retain_samples": len(retain_qa),
        "forget_size_requested": args.forget_size,
        "retain_size_requested": args.retain_size,
        "forget_answer_sources": {
            "predict": forget_predict_count,
            "label": forget_label_count
        },
        "retain_answer_sources": {
            "predict": retain_predict_count,
            "label": retain_label_count
        },
        "seed": args.seed,
        "description": {
            "forget_set": "exec_match=false ， predict  answer ()",
            "retain_set": "exec_match=true  predict， false  label "
        }
    }
    stats_file = output_dir / "split_stats.json"
    save_json(stats, stats_file)
    
    print(f"\n{'='*80}")
    print("✅ !")
    print(f"{'='*80}")
    print(f"  - Forget set: {forget_file}")
    print(f"  - Retain set: {retain_file}")
    print(f"  - : {stats_file}")
    print()
    print(":")
    print(f"  1️Forget:  exec_match=false  {args.forget_size} ， predict ()")
    print(f"  2️Retain:  exec_match=true  {args.retain_size} ， predict ()")
    print(f"  3️ retain ， false  label  ()")
    print()


if __name__ == "__main__":
    main()
