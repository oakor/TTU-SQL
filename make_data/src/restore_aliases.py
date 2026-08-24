#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SQL，
"""

import os
import json
import re
from typing import Dict, List, Tuple


def extract_aliases_from_sql(sql: str) -> Dict[str, str]:
    """
    SQL
    : FROM observations o JOIN sites s ON ... -> {'observations': 'o', 'sites': 's'}
    """
    aliases = {}
    
    #  FROM table_name alias  FROM table_name AS alias 
    #  JOIN table_name alias  JOIN table_name AS alias 
    pattern = r'\b(FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\b'
    matches = re.findall(pattern, sql, re.IGNORECASE)
    
    for clause, table_name, alias_name in matches:
        # 
        if table_name not in aliases:
            aliases[table_name] = alias_name
    
    return aliases


def restore_aliases_in_sql(sql: str, aliases: Dict[str, str]) -> str:
    """
    SQL
    : observations.object_name -> o.object_name (oobservations)
    """
    result = sql
    
    # ，
    sorted_aliases = sorted(aliases.items(), key=lambda x: len(x[0]), reverse=True)
    
    for table_name, alias_name in sorted_aliases:
        #  table_name.column_name  alias_name.column_name
        # 
        pattern = r'\b' + re.escape(table_name) + r'\.(\w+)'
        replacement = f'{alias_name}.\\1'
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result


def process_jsonl_file(input_path: str, output_path: str):
    """
    JSONL，SQL
    """
    print(f"Processing file: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    processed_lines = []
    
    for line_num, line in enumerate(lines, 1):
        try:
            data = json.loads(line.strip())
            
            # SQL
            original_predict = data.get('predict', '')
            original_label = data.get('label', '')
            
            # SQL
            normalized_predict = data.get('normalized_predict', '')
            normalized_label = data.get('normalized_label', '')
            
            # SQL
            predict_aliases = extract_aliases_from_sql(original_predict)
            label_aliases = extract_aliases_from_sql(original_label)
            
            # SQL
            restored_predict_from_normalized = restore_aliases_in_sql(normalized_predict, predict_aliases)
            restored_label_from_normalized = restore_aliases_in_sql(normalized_label, label_aliases)
            
            # 
            data['restored_normalized_predict'] = restored_predict_from_normalized
            data['restored_normalized_label'] = restored_label_from_normalized
            
            processed_lines.append(json.dumps(data, ensure_ascii=False))
            
            if line_num % 10 == 0:
                print(f"Processed {line_num} lines...")
                
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON at line {line_num}: {e}")
            continue
        except Exception as e:
            print(f"Error processing line {line_num}: {e}")
            continue
    
    # 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for line in processed_lines:
            outfile.write(line + '\n')
    
    print(f"Processed {len(processed_lines)} lines. Output saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Restore aliases in normalized SQL queries")
    parser.add_argument("--input_path", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSONL file path")
    
    args = parser.parse_args()
    
    process_jsonl_file(args.input_path, args.output_path)


if __name__ == "__main__":
    main()