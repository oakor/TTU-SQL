#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

NPO
omnisql_npospider
"""

import os, re
import sys
import json
import sqlite3
import argparse
import traceback
from tqdm import tqdm

# NLTK
import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print(" NLTK punkt_tab ...")
    nltk.download('punkt_tab', quiet=True)

# 
sys.path.append('../../evaluation/spider-master')
from process_sql import get_schema, Schema, get_sql
from evaluation import build_foreign_key_map_from_json, rebuild_sql_val, rebuild_sql_col, build_valid_col_units


def load_test_data(test_file):
    """"""
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_predictions(pred_file):
    """"""
    predictions = []
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pred_data = json.loads(line.strip())
                predictions.append(pred_data)
    return predictions


def load_train_data(train_files, data_format='omnisql'):
    """db_id"""
    train_data = {}
    for train_file in train_files:
        if os.path.exists(train_file):
            with open(train_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for idx, item in enumerate(data):
                    # SQL
                    if data_format == 'omnisql':
                        train_data[idx] = item['db_id']
                        continue
                    elif data_format == 'spider':
                        sql_field = 'query'
                    else:
                        sql_field = 'sql'
                    
                    # SQLkey
                    if sql_field in item and 'db_id' in item:
                        sql = item.get(sql_field, '').replace("\n", " ")
                        # 
                        sql = re.sub(r'\s+', ' ', sql).strip().replace("\n", " ")
                        train_data[sql] = item['db_id']
    return train_data


def eval_exec_match(db, p_str, g_str, pred, gold):
    """
    ：SQLSQL
     evaluation.py  eval_exec_match 
    : (is_match, error_msg, p_res, g_res)
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    
    try:
        cursor.execute(p_str)
        p_res = cursor.fetchall()
    except Exception as e:
        conn.close()
        return False, f"SQL: {str(e)}", None, None
    
    try:
        cursor.execute(g_str)
        q_res = cursor.fetchall()
    except Exception as e:
        conn.close()
        return False, f"SQL: {str(e)}", p_res, None
    
    conn.close()
    
    # 
    # 
    def normalize_result(res):
        """："""
        if not res:
            return []
        # 
        normalized = []
        for row in res:
            # （）
            if isinstance(row, (list, tuple)):
                normalized.append(tuple(row))
            else:
                normalized.append((row,))
        # 
        try:
            normalized.sort()
        except TypeError:
            # （None），
            pass
        return normalized
    
    p_normalized = normalize_result(p_res)
    q_normalized = normalize_result(q_res)
    
    is_match = p_normalized == q_normalized
    
    if not is_match:
        return False, f" - : {p_res}, : {q_res}", p_res, q_res
    
    return True, None, p_res, q_res


def extract_schema_and_nl_from_prompt(prompt: str) -> tuple[str, str]:
    """
    promptschemanl
    ：
    - split('\n\n')1schema
    - split('\n\n')2，<|eot_id|><|start_header_id|>assistant<|end_header_id|>nl
    """
    try:
        parts = prompt.split('\n\n')
        if len(parts) >= 3:
            schema = parts[1].split("\n###Input:\nSchema:\n")[-1]
            nl = parts[2].replace('\n###Response:<|eot_id|><|start_header_id|>assistant<|end_header_id|>', '').strip()
            return schema, nl
        else:
            return "", ""
    except Exception as e:
        return "", ""


def process_nl_schema(str_a):
    """NLSchema"""
    return str_a.split("\n\n")[0].split("###Input:\nSchema:\n")[-1], str_a.split("\n\n")[1].split("\n###Response:")[0]


def evaluate(pred_file, test_file=None, output_dir=None, data_format='omnisql'):
    """
    
    
    Args:
        pred_file: 
        test_file: (,data_format)
        output_dir: (,pred_file)
        data_format:  ('omnisql'  'spider')
    """
    # data_format
    if data_format == 'omnisql':
        # OmniSQL
        train_files = [
            '../data/data_50000.json'
        ]
        db_dir = '../data/databases'
        table_file = '../data/tables.json'
        
        # test_file，
        if test_file is None:
            test_file = '../data/omnisql/omniSQL_50000_alpaca.json'
            
    elif data_format == 'spider':
        # Spider
        train_files = [
            '../data/spider/train_spider.json'
        ]
        db_dir = '../data/spider/database'
        table_file = '../data/spider/tables.json'
        
        # test_file，
        if test_file is None:
            test_file = '../data/spider/spider_dev.json'
            
    else:
        raise ValueError(f"data_format: {data_format}")

    
    # 
    if output_dir is None:
        output_dir = os.path.dirname(pred_file)
    
    print("="*80)
    print(f"")
    print("="*80)
    print(f": {pred_file}")
    print(f": {test_file}")
    print(f": {data_format}")
    print(f": {output_dir}")
    print("="*80)
    
    print("\n...")
    test_data = load_test_data(test_file)
    predictions = load_predictions(pred_file)
    train_data = load_train_data(train_files, data_format=data_format)
    # 
    print("...")
    kmaps = build_foreign_key_map_from_json(table_file)
    
    # 
    if len(test_data) != len(predictions):
        print(f":  ({len(test_data)})  ({len(predictions)}) !")
        print(f": {test_file}")
        print(f": {pred_file}")
        print(f"。。")
        # ，
        raise ValueError(f": test_data={len(test_data)}, predictions={len(predictions)}")
    
    print(f": {len(test_data)}")
    
    # 
    total = 0
    exec_correct = 0
    no_db_id = 0
    exec_error = 0
    parse_error = 0
    
    # （predictions，exec_matchindex）
    enhanced_predictions = []
    
    for i, (test_item, pred_item) in enumerate(tqdm(zip(test_data, predictions), total=len(test_data))):
        total += 1
        
        # schema、nlSQL
        # alpaca(input/output)QA(question/answer)
        if 'input' in test_item and 'output' in test_item:
            # alpaca
            schema_str, nl = process_nl_schema(test_item["input"])
            g_str = test_item['output']
        elif 'question' in test_item and 'answer' in test_item:
            # QA (omnisql_npo: forget.json/retain.json)
            # questioninstruction+schema+nl
            question_text = test_item['question']
            # questionschemanl
            try:
                if '###Input:\nSchema:\n' in question_text:
                    parts = question_text.split('###Input:\nSchema:\n')
                    if len(parts) > 1:
                        schema_and_rest = parts[1]
                        # schemanl
                        if '\n\n' in schema_and_rest:
                            schema_str = schema_and_rest.split('\n\n')[0]
                            nl = schema_and_rest.split('\n\n')[1].replace('###Response:', '').strip()
                        else:
                            schema_str = schema_and_rest
                            nl = ""
                    else:
                        schema_str = ""
                        nl = ""
                else:
                    schema_str = ""
                    nl = ""
            except Exception as e:
                schema_str = ""
                nl = ""
            g_str = test_item['answer']
        else:
            print(f"\n: {i}，")
            continue
            
        # SQL
        p_str = pred_item['predict']
        
        #  db_id
        db_id = train_data.get(test_item["id"])
        
        # （）
        enhanced_pred = pred_item.copy()
        enhanced_pred['index'] = i
        enhanced_pred['exec_match'] = False  # 
        enhanced_pred['error'] = None  # error
        enhanced_pred['gold_sql'] = g_str  # SQL
        
        # promptschemanl
        prompt_schema, prompt_nl = extract_schema_and_nl_from_prompt(pred_item.get('prompt', ''))
        enhanced_pred['schema'] = prompt_schema
        enhanced_pred['nl'] = prompt_nl
        
        if db_id is None:
            no_db_id += 1
            enhanced_pred['error'] = "db_id"
            enhanced_predictions.append(enhanced_pred)
            continue
        
        # 
        db_path = os.path.join(db_dir, db_id, db_id + ".sqlite")
        enhanced_pred['db_id'] = db_id  # db_id
        
        if not os.path.exists(db_path):
            no_db_id += 1
            enhanced_pred['error'] = f": {db_path}"
            enhanced_predictions.append(enhanced_pred)
            continue
        
        # schema
        try:
            schema = Schema(get_schema(db_path))
        except Exception as e:
            exec_error += 1
            enhanced_pred['error'] = f"schema: {str(e)}"
            enhanced_predictions.append(enhanced_pred)
            continue
        
        # SQL
        try:
            g_sql = get_sql(schema, g_str)
        except Exception as e:
            parse_error += 1
            enhanced_pred['error'] = f"SQL: {str(e)}"
            enhanced_predictions.append(enhanced_pred)
            continue
        
        # SQL
        try:
            p_sql = get_sql(schema, p_str)
        except Exception as e:
            # SQL，SQL
            p_sql = {
                "except": None,
                "from": {
                    "conds": [],
                    "table_units": []
                },
                "groupBy": [],
                "having": [],
                "intersect": None,
                "limit": None,
                "orderBy": [],
                "select": [
                    False,
                    []
                ],
                "union": None,
                "where": []
            }
            parse_error += 1
            enhanced_pred['error'] = f"SQL: {str(e)}"
        
        # SQL
        try:
            kmap = kmaps[db_id]
            g_valid_col_units = build_valid_col_units(g_sql['from']['table_units'], schema)
            g_sql = rebuild_sql_val(g_sql)
            g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)
            p_valid_col_units = build_valid_col_units(p_sql['from']['table_units'], schema)
            p_sql = rebuild_sql_val(p_sql)
            p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)
        except Exception as e:
            exec_error += 1
            enhanced_pred['error'] = f"SQL: {str(e)}"
            enhanced_predictions.append(enhanced_pred)
            continue
        
        # 
        try:
            # SQL（evaluation.py）
            exec_match, error_msg, p_res, g_res = eval_exec_match(db_path, p_str, g_str, p_sql, g_sql)
            if p_res is not None:
                enhanced_pred['pred_result'] = str(p_res)
            if g_res is not None:
                enhanced_pred['gold_result'] = str(g_res)
            
            if exec_match:
                exec_correct += 1
            
            # exec_matcherror
            enhanced_pred['exec_match'] = exec_match
            if error_msg:
                enhanced_pred['error'] = error_msg
            enhanced_predictions.append(enhanced_pred)
        except Exception as e:
            exec_error += 1
            enhanced_pred['error'] = f": {str(e)}\n{traceback.format_exc()}"
            enhanced_predictions.append(enhanced_pred)
    
    # 
    print("\n" + "=" * 80)
    print("")
    print("=" * 80)
    print(f": {total}")
    print(f": {exec_correct}")
    print(f": {exec_correct / total * 100:.2f}%")
    print(f"db_id: {no_db_id}")
    print(f"SQL: {parse_error}")
    print(f": {exec_error}")
    print("=" * 80)
    
    # （JSON，+exec_match、index、schema、nl）
    base_name = os.path.splitext(os.path.basename(pred_file))[0]
    enhanced_pred_file = os.path.join(output_dir, f'{base_name}_with_exec.json')
    with open(enhanced_pred_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_predictions, f, indent=2, ensure_ascii=False)
    
    print(f"\n: {enhanced_pred_file}")
    
    # 
    summary_file = os.path.join(output_dir, f'{base_name}_exec_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write(f": {total}\n")
        f.write(f": {exec_correct}\n")
        f.write(f": {exec_correct / total * 100:.2f}%\n")
        f.write(f"db_id: {no_db_id}\n")
        f.write(f"SQL: {parse_error}\n")
        f.write(f": {exec_error}\n")
        f.write("=" * 80 + "\n")
    
    print(f": {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="NPO")
    
    # 
    parser.add_argument("--pred_file", type=str, required=True,
                        help=" (.jsonl)")
    
    # 
    parser.add_argument("--test_file", type=str, default=None,
                        help="（，data_format）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="（，pred_file）")
    parser.add_argument("--data_format", type=str, default="omnisql",
                        choices=["omnisql", "spider"],
                        help=": omnisql  spider")
    
    args = parser.parse_args()
    
    evaluate(
        pred_file=args.pred_file,
        test_file=args.test_file,
        output_dir=args.output_dir,
        data_format=args.data_format
    )


if __name__ == "__main__":
    main()
