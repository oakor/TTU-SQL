import json
import os
import random

def find_divergence_idx(s1, s2):
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return i
    if len(s1) != len(s2):
        return min_len
    return -1

def extract_question_content(prompt):
    # Extract the content between user tags
    user_start = "<|start_header_id|>user<|end_header_id|>\n\n"
    eot = "<|eot_id|>"
    if user_start in prompt:
        start_idx = prompt.find(user_start) + len(user_start)
        # Look for the end of the user message
        end_idx = prompt.find(eot, start_idx)
        if end_idx != -1:
            return prompt[start_idx:end_idx].strip()
    return prompt

def process():
    source_file = "generated_predictions_with_exec_unified.json"
    output_dir = "../token_level_data_process"
    item_count = 10000
    seed = 42
    
    if not os.path.exists(source_file):
        print(f"Error: Source file {source_file} not found.")
        return

    with open(source_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    forget_token_first = []
    forget_token_whole = []
    
    for i, item in enumerate(data):
        # Filter for exec_match == false
        if item.get("exec_match") is False:
            predict = item.get("predict", "")
            label = item.get("label", "")
            
            diff_idx = find_divergence_idx(predict, label)
            if diff_idx == -1:
                # If they are identical (should not happen if exec_match is false), forget the whole thing
                forget_tail = predict
            else:
                forget_tail = predict[diff_idx:]
            
            # Extract question content to avoid double templating in NPO trainer
            question = extract_question_content(item.get("prompt", ""))
            
            base_item = {
                "id": str(i),
                "question": question,
                "answer": predict,
                "db_id": item.get("db_id", ""),
                "answer_source": "predict",
                "original_exec_match": item.get("exec_match"),
                "original_error": item.get("error")
            }
            
            # Logic 1: Only forget the first token at divergence
            item_first = base_item.copy()
            item_first["forget_tail"] = forget_tail
            item_first["forget_mode"] = "First"
            # forget_part is not used in First mode but we set it to the first word for clarity
            item_first["forget_part"] = forget_tail.split()[0] if forget_tail.split() else (forget_tail[:1] if forget_tail else "")
            forget_token_first.append(item_first)
            
            # Logic 2: Forget the entire suffix from divergence
            item_whole = base_item.copy()
            item_whole["forget_tail"] = forget_tail
            item_whole["forget_mode"] = "Whole"
            item_whole["forget_part"] = forget_tail
            forget_token_whole.append(item_whole)
    
    random.seed(seed)
    random_idx = random.sample(list(range(len(forget_token_first))), item_count)
    forget_token_first = [forget_token_first[i] for i in random_idx]
    forget_token_whole = [forget_token_whole[i] for i in random_idx]
    # Save files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(os.path.join(output_dir, "forget_token_first.json"), 'w', encoding='utf-8') as f:
        json.dump(forget_token_first, f, indent=4, ensure_ascii=False)
        
    with open(os.path.join(output_dir, "forget_token_whole.json"), 'w', encoding='utf-8') as f:
        json.dump(forget_token_whole, f, indent=4, ensure_ascii=False)
    
    print(f"Processed {len(forget_token_first)} incorrect predictions.")
    print(f"Saved to {output_dir}")

if __name__ == "__main__":
    process()
