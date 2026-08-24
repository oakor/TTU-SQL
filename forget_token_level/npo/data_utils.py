#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

 SFT 
"""

import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List
from dataclasses import dataclass
from tqdm import tqdm


class QADataset(Dataset):
    """
    QA 
    : 
    {
        "id": "xxx", 
        "question": "...", 
        "answer": "SELECT name FROM users WHERE age > 18",
        "forget_tail": "users WHERE age > 18",  # answer（）
        "forget_part": "users",  # Firsttoken，Whole
        "forget_mode": "First"  # "First"=token, "Whole"=forget_part
    }
    
     forget_tail， answer
    """
    
    def __init__(self, file_path, tokenizer, max_length=3200):
        self.data = self.load_data(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "question": item["question"],
            "answer": item["answer"],
            "forget_tail": item.get("forget_tail", None),  # 
            "forget_part": item.get("forget_part", None),  # 
            "forget_mode": item.get("forget_mode", "First"),  # "First"  "Whole"
            "id": item.get("id", str(idx))
        }


class ForgetRetainDataset(Dataset):
    """
    Forget-Retain 
     forget  retain ，
    """
    
    def __init__(self, forget_dataset, retain_dataset, anchor="forget"):
        """
        Args:
            forget_dataset: 
            retain_dataset: 
            anchor: "forget"  "retain"，
        """
        self.forget = forget_dataset
        self.retain = retain_dataset
        self.anchor = anchor
    
    def __len__(self):
        if self.anchor == "forget":
            return len(self.forget)
        else:
            return len(self.retain)
    
    def __getitem__(self, idx):
        """
         forget-retain 
        """
        item = {}
        
        if self.anchor == "forget":
            item["forget"] = self.forget[idx]
            #  retain 
            retain_idx = torch.randint(0, len(self.retain), (1,)).item()
            item["retain"] = self.retain[retain_idx]
        else:
            item["retain"] = self.retain[idx]
            #  forget 
            forget_idx = torch.randint(0, len(self.forget), (1,)).item()
            item["forget"] = self.forget[forget_idx]
        
        return item


@dataclass
class NPODataCollator:
    """
    NPO  collator
     forget-retain ， SFT 
     token  (forget_mask)
    """
    
    tokenizer: any
    max_length: int = 3200
    debug: bool = True  # 
    
    def apply_chat_template(self, question, answer):
        """
         llama3 chat 
         SFT 
        """
        # llama3 
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        
        #  tokenizer  chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        return text
    
    def find_forget_token_positions(self, input_ids, answer, forget_tail, forget_part, forget_mode, assistant_start_pos, is_forget_sample=True):
        """
         forget_tail  token 
        
        Args:
            input_ids: tokenize  token ids
            answer: 
            forget_tail: answer（）
            forget_part: Firsttoken，Whole
            forget_mode: "First"  "Whole"
            assistant_start_pos: assistant  token 
            is_forget_sample: （）
        
        Returns:
            forget_mask:  input_ids ，1 ，0 
        """
        forget_mask = [0] * len(input_ids)
        
        if forget_tail is None:
            #  forget_tail， assistant 
            for i in range(assistant_start_pos, len(input_ids)):
                forget_mask[i] = 1
            if self.debug and is_forget_sample:
                print(f"[DEBUG] No forget_tail specified, forgetting entire answer")
                print(f"  - Forget range: [{assistant_start_pos}:{len(input_ids)}] (total {len(input_ids)-assistant_start_pos} tokens)")
            return forget_mask
        
        # ： assistant_start_pos 
        #  \n\n 
        actual_content_start = assistant_start_pos
        for i in range(assistant_start_pos, min(assistant_start_pos + 10, len(input_ids))):
            decoded = self.tokenizer.decode([input_ids[i]])
            if decoded.strip(): #  token
                actual_content_start = i
                break
        
        #  forget_tail （）
        #  forget_tail  answer 
        tail_start_char = answer.find(forget_tail)
        if tail_start_char == -1 or not answer.endswith(forget_tail):
            if self.debug and is_forget_sample:
                print(f"[DEBUG] WARNING: forget_tail '{forget_tail[:30]}...' not found as suffix in answer")
                print(f"[DEBUG] Falling back to full answer forgetting")
            for i in range(actual_content_start, len(input_ids)):
                forget_mask[i] = 1
            return forget_mask
        
        #  prefix （forget_tail ）
        prefix = answer[:tail_start_char]
        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False) if prefix else []
        
        # forget_tail  input_ids 
        forget_start_pos = actual_content_start + len(prefix_tokens)
        
        if forget_mode == "First":
            #  token
            if forget_start_pos < len(input_ids):
                forget_mask[forget_start_pos] = 1
            
            if self.debug and is_forget_sample:
                first_token_id = input_ids[forget_start_pos] if forget_start_pos < len(input_ids) else None
                first_token_text = self.tokenizer.decode([first_token_id]) if first_token_id else "N/A"
                print(f"[DEBUG] Token-level forget (First mode):")
                print(f"  - Answer: '{answer[:50]}...'")
                print(f"  - forget_tail starts at char {tail_start_char}")
                print(f"  - Content token offset: {actual_content_start - assistant_start_pos} tokens")
                print(f"  - Forget position: {forget_start_pos} (1 token)")
                print(f"  - Targeted token: '{first_token_text}' (id={first_token_id})")
                
        elif forget_mode == "Whole":
            #  forget_part
            if forget_part is None:
                if self.debug and is_forget_sample:
                    print(f"[DEBUG] WARNING: forget_mode='Whole' but forget_part is None")
                forget_part = forget_tail  # fallback
            
            forget_part_tokens = self.tokenizer.encode(forget_part, add_special_tokens=False)
            forget_end_pos = forget_start_pos + len(forget_part_tokens)
            forget_end_pos = min(forget_end_pos, len(input_ids))
            
            for i in range(forget_start_pos, forget_end_pos):
                forget_mask[i] = 1
            
            if self.debug and is_forget_sample:
                print(f"[DEBUG] Token-level forget (Whole mode):")
                print(f"  - Answer: '{answer[:50]}...'")
                print(f"  - forget_tail starts at char {tail_start_char}")
                print(f"  - forget_part: '{forget_part[:30]}...' ({len(forget_part_tokens)} tokens)")
                print(f"  - Forget range: [{forget_start_pos}:{forget_end_pos}] ({forget_end_pos-forget_start_pos} tokens)")
        else:
            if self.debug and is_forget_sample:
                print(f"[DEBUG] WARNING: Unknown forget_mode '{forget_mode}', using 'First'")
            if forget_start_pos < len(input_ids):
                forget_mask[forget_start_pos] = 1
        
        if self.debug and is_forget_sample:
            print(f"  - Total sequence: {len(input_ids)} tokens, forget ratio: {sum(forget_mask)}/{len(input_ids)} = {sum(forget_mask)/len(input_ids)*100:.1f}%")
        
        return forget_mask
    
    def tokenize_sample(self, question, answer, forget_tail=None, forget_part=None, forget_mode="First", is_forget_sample=True):
        """
         tokenize
        
        Args:
            question: 
            answer: 
            forget_tail: （）
            forget_part: 
            forget_mode: "First"  "Whole"
            is_forget_sample: 
        
        Returns:
            dict with input_ids, attention_mask, labels, forget_mask
        """
        #  chat 
        full_text = self.apply_chat_template(question, answer)
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        
        #  labels
        assistant_header = "<|start_header_id|>assistant<|end_header_id|>"
        assistant_tokens = self.tokenizer.encode(assistant_header, add_special_tokens=False)
        
        labels = input_ids.copy()
        assistant_start_pos = len(input_ids)  # 
        
        #  assistant 
        for i in range(len(input_ids) - len(assistant_tokens) + 1):
            if input_ids[i:i+len(assistant_tokens)] == assistant_tokens:
                labels[:i+len(assistant_tokens)] = [-100] * (i + len(assistant_tokens))
                assistant_start_pos = i + len(assistant_tokens)
                break
        
        #  forget_mask
        forget_mask = self.find_forget_token_positions(
            input_ids, answer, forget_tail, forget_part, forget_mode, assistant_start_pos, is_forget_sample
        )
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "forget_mask": forget_mask
        }
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate  batch 
        
        Args:
            features: List of dicts, each contains "forget" and "retain" items
        
        Returns:
            Dict with "forget" and "retain" batches, each containing forget_mask
        """
        forget_batch = []
        retain_batch = []
        
        for idx, feature in tqdm(enumerate(features)):
            if self.debug and idx == 0:
                print(f"\n[DEBUG] Processing batch sample {idx}:")
            
            #  forget sample
            forget_item = feature["forget"]
            forget_encoded = self.tokenize_sample(
                forget_item["question"],
                forget_item["answer"],
                forget_item.get("forget_tail", None),
                forget_item.get("forget_part", None),
                forget_item.get("forget_mode", "First"),
                is_forget_sample=True
            )
            forget_batch.append(forget_encoded)
            
            #  retain sample (retain  token )
            retain_item = feature["retain"]
            retain_encoded = self.tokenize_sample(
                retain_item["question"],
                retain_item["answer"],
                None, None, "First",
                is_forget_sample=False
            )
            retain_batch.append(retain_encoded)
        
        # Padding and convert to tensors
        def pad_batch(batch, include_forget_mask=True):
            max_len = min(max(len(x["input_ids"]) for x in batch), self.max_length)
            
            input_ids = []
            attention_mask = []
            labels = []
            forget_masks = []
            
            for item in batch:
                pad_len = max_len - len(item["input_ids"])
                
                padded_input_ids = item["input_ids"] + [self.tokenizer.pad_token_id] * pad_len
                padded_attention_mask = item["attention_mask"] + [0] * pad_len
                padded_labels = item["labels"] + [-100] * pad_len
                padded_forget_mask = item["forget_mask"] + [0] * pad_len
                
                input_ids.append(padded_input_ids)
                attention_mask.append(padded_attention_mask)
                labels.append(padded_labels)
                forget_masks.append(padded_forget_mask)
            
            result = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
            
            if include_forget_mask:
                result["forget_mask"] = torch.tensor(forget_masks, dtype=torch.long)
            
            return result
        
        return {
            "forget": pad_batch(forget_batch, include_forget_mask=True),
            "retain": pad_batch(retain_batch, include_forget_mask=False)
        }
