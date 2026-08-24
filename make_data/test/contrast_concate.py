import json, jsonlines
from Llama3_Embedder import Llama3Embedding
from tqdm import tqdm

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_divergence(file_path):
    with jsonlines.open(file_path) as reader:
        divergence = [obj for obj in reader]
    return divergence

exe_path = "../output/BIRD_deepseek_llama_8B/bird_train/omnisql_continue_train_epoch1to15/predict/checkpoint-1667/predict_omnisql/generated_predictions_with_exec_unified.json"

divergence_path = "../output/bird_deepseek_llama/generated_predictions_with_divergence.jsonl"

exe_data = load_data(exe_path)
divergence_data = load_divergence(divergence_path)

exec_false_data = []
idx = 0
for exe, diver in tqdm(zip(exe_data, divergence_data)):
    # if exe["gold_sql"] != diver["label"]:
    #     print(f"Label mismatch at index {idx}")
    assert exe["gold_sql"] == diver["label"]
    if exe["exec_match"] is not True:
        diver["db_id"] = exe["db_id"]
        diver["index"] = exe["index"]
        exec_false_data.append(diver)
    # idx += 1

def get_class_1(datas):
    SQL_KEYWORDS = {
    "WHERE", "ON", "USING", "GROUP", "ORDER", "HAVING", "LIMIT", "OFFSET", "FETCH",
    "UNION", "INTERSECT", "EXCEPT",
    "INNER", "LEFT", "RIGHT", "FULL", "CROSS", "OUTER", "NATURAL",
    "JOIN", "FROM", "AS",
    "AND", "OR", "NOT", "IN", "IS", "NULL", "LIKE", "BETWEEN",
    "SELECT", "DISTINCT",
    }
    output = []
    for data in tqdm(datas):
        if not data["suffix"].split():
            data["suffix"] = "<|eot_id|>"
        if data["suffix"].split()[0] in SQL_KEYWORDS:
            output.append(data)
    return output

def get_class_2(datas):
    # 2：
    # 1. labelprefixlabel_suffix
    # 2. top5 token，tokenlabel_suffix
    # 3. /（）
    # 4. token，dataoutput
    output = []
    for data in tqdm(datas):
        found_match = False
        label = data.get('label', '')
        prefix = data.get('prefix', '')
        top5_tokens = data.get('logits_top5', [{}])[0].get('top5_tokens', [])

        if label.startswith(prefix):
            label_suffix = label[len(prefix):].strip()  # 
        else:
            label_tokens = label.split()
            prefix_tokens = prefix.split()
            label_suffix = " ".join(label_tokens[len(prefix_tokens):]).strip()
        
        if not label_suffix:
            label_suffix = "<|eot_id|>"
        # top5 token，
        for token in top5_tokens:
            # token：+，token
            clean_token = str(token).strip()
            if not clean_token:
                continue
            
            # ：1.tokenlabel_suffix  2./（）
            if label_suffix.startswith(clean_token):
                # token
                rest_part = label_suffix[len(clean_token):]
                # 2： → ； → /（isalnum()）
                if not rest_part or not rest_part[0].isalnum():
                    found_match = True  # token
                    break  # token
        
        # token，data（append）
        if not found_match:
            output.append(data)
    
    return output

def get_class_3(datas):
    # 2：
    # 1. labelprefixlabel_suffix
    # 2. top5 token，tokenlabel_suffix
    # 3. /（）
    # 4. token，tokentoken，(80)output
    # 
    print("Initializing Llama3.2 1B embedding model...")
    embedder = Llama3Embedding()
    output = []
    for data in tqdm(datas):
        found_match = False
        label = data.get('label', '')
        prefix = data.get('prefix', '')
        top5_tokens = data.get('logits_top5', [{}])[0].get('top5_tokens', [])

        if label.startswith(prefix):
            label_suffix = label[len(prefix):].strip()  # 
        else:
            label_tokens = label.split()
            prefix_tokens = prefix.split()
            label_suffix = " ".join(label_tokens[len(prefix_tokens):]).strip()

        if not label_suffix:
            label_suffix = "<|eot_id|>"
        # top5 token，
        for token in top5_tokens:
            # token：+，token
            clean_token = str(token).strip()
            if not clean_token:
                continue
            
            # ：1.tokenlabel_suffix  2./（）
            if label_suffix.startswith(clean_token):
                # token
                rest_part = label_suffix[len(clean_token):]
                # 2： → ； → /（isalnum()）
                if not rest_part or not rest_part[0].isalnum():
                    similarity = embedder.similarity(clean_token, data.get('logits_top5', [{}])[0].get("generated_token", '').strip())
                    if similarity < 0.8:
                        found_match = True  # token
                        break  # token
        
        # token，data（append）
        if found_match:
            output.append(data)
    
    return output

print("Getting class 1 data...")
class_1_datas = get_class_1(exec_false_data)
json.dump(class_1_datas, open("../output/bird_deepseek_llama/to_forget/forget_class_1.json", 'w'), indent=4)
print("Getting class 2 data...")
class_2_datas = get_class_2(exec_false_data)
json.dump(class_2_datas, open("../output/bird_deepseek_llama/to_forget/forget_class_2.json", 'w'), indent=4)
print("Getting class 3 data...")
class_3_datas = get_class_3(exec_false_data)
json.dump(class_3_datas, open("../output/bird_deepseek_llama/to_forget/forget_class_3.json", 'w'), indent=4)

output_all = []
for item in class_1_datas + class_2_datas + class_3_datas:
    if item not in output_all:
        output_all.append(item)

json.dump(output_all, open("../output/bird_deepseek_llama/to_forget/forget_class_all.json", 'w'), indent=4)
