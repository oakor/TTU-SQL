#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Spider ，：
- prompt / predict / label
-  predict  label（aliases ）
-  ****  top-5  token 
-  prefix（ predict ， token ）
-  suffix（ predict ， prefix ）
- divergence_meta（）：normalized  token  predict 

（A）：
1)  predict / label  aliases （、 SELECT ）。
2)  normalized_predict vs normalized_label  token ， token（ span）。
3)  difflib.SequenceMatcher ， normalized_predict -> original_predict 。
4)  normalized  token  original predict， predict  \\S+ token ，
    prefix/suffix（ normalize  token  idx ）。
5)  prefix  step， step  top-5 logits 。

：
- step  split()（ subword token  step ）， tokenizer  token_id 。
-  extract_aliases_from_sql： WHERE/ON （ FROM health_data WHERE...）。
"""

import os
import json
import argparse
import re
import difflib
from typing import Dict, List, Any, Optional, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from tqdm import tqdm


# ====================  ====================

def load_spider_dataset(dataset_dir: str, dataset_name: str = "spider_dev.json") -> List[Dict[str, Any]]:
    """ Spider ， LLaMA-Factory 。"""
    dataset_path = os.path.join(dataset_dir, dataset_name)
    print(f"Loading dataset from: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")
    return data


# ==================== SQL / ====================

SQL_KEYWORDS = {
    "WHERE", "ON", "USING", "GROUP", "ORDER", "HAVING", "LIMIT", "OFFSET", "FETCH",
    "UNION", "INTERSECT", "EXCEPT",
    "INNER", "LEFT", "RIGHT", "FULL", "CROSS", "OUTER", "NATURAL",
    "JOIN", "FROM", "AS",
    "AND", "OR", "NOT", "IN", "IS", "NULL", "LIKE", "BETWEEN",
    "SELECT", "DISTINCT",
}

TYPE_KEYWORDS = {
    "INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT",
    "NUMERIC", "DECIMAL", "FLOAT", "REAL", "DOUBLE",
    "CHAR", "NCHAR", "VARCHAR", "NVARCHAR", "TEXT", "STRING",
    "DATE", "TIME", "TIMESTAMP", "DATETIME",
    "BOOLEAN", "BOOL",
}


def extract_aliases_from_sql(sql: str) -> Dict[str, str]:
    """
    SQL（table_name -> alias）
    : FROM observations o JOIN sites s ON ... -> {'observations': 'o', 'sites': 's'}

    ： WHERE/ON 
    （：FROM health_data WHERE ...  WHERE  alias）
    """
    aliases: Dict[str, str] = {}

    pattern = r'\b(FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\b'
    matches = re.findall(pattern, sql, re.IGNORECASE)

    for _, table_name, alias_name in matches:
        if alias_name.upper() in SQL_KEYWORDS:
            continue
        if table_name not in aliases:
            aliases[table_name] = alias_name
    return aliases


def normalize_sql_remove_aliases(sql: str) -> str:
    """
     SQL “ + (SELECT ... AS alias)”
    1)  FROM/JOIN table [AS] alias， alias->table （）
    2)  alias.column  table.column
    3)  FROM/JOIN  alias ：FROM table alias -> FROM table
    4)  SELECT ：... AS alias（ CAST  AS TYPE）
    """
    s = sql

    tbl_alias_re = re.compile(
        r"\b(FROM|JOIN)\s+([A-Za-z_][\w$]*(?:\.[A-Za-z_][\w$]*)*)\s+(?:AS\s+)?([A-Za-z_][\w$]*)\b",
        re.IGNORECASE
    )

    alias_to_table: Dict[str, str] = {}
    for m in tbl_alias_re.finditer(s):
        table = m.group(2)
        alias = m.group(3)
        if alias.upper() in SQL_KEYWORDS:
            continue
        alias_to_table[alias] = table

    # alias.col -> table.col
    for alias, table in alias_to_table.items():
        s = re.sub(rf"\b{re.escape(alias)}\.", f"{table}.", s, flags=re.IGNORECASE)

    # drop FROM/JOIN alias defs
    def drop_tbl_alias(m: re.Match) -> str:
        kw, table, alias = m.group(1), m.group(2), m.group(3)
        if alias.upper() in SQL_KEYWORDS:
            return m.group(0)
        return f"{kw} {table}"

    s = tbl_alias_re.sub(drop_tbl_alias, s)

    # drop SELECT col aliases: ... AS alias (avoid CAST AS TYPE)
    m = re.search(r"\bSELECT\b(.*?)\bFROM\b", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        select_part = m.group(1)

        def drop_col_alias(mm: re.Match) -> str:
            alias = mm.group(1)
            if alias.upper() in TYPE_KEYWORDS:
                return mm.group(0)
            return ""

        select_part2 = re.sub(
            r"\s+\bAS\b\s+([A-Za-z_][\w$]*)\b",
            drop_col_alias,
            select_part,
            flags=re.IGNORECASE
        )
        s = s[:m.start(1)] + select_part2 + s[m.end(1):]

    s = re.sub(r"\s+", " ", s).strip()
    return s


def restore_aliases_in_sql(sql: str, aliases: Dict[str, str]) -> str:
    """
    SQL
    : observations.object_name -> o.object_name (oobservations)
    """
    result = sql
    sorted_aliases = sorted(aliases.items(), key=lambda x: len(x[0]), reverse=True)

    for table_name, alias_name in sorted_aliases:
        pattern = r'\b' + re.escape(table_name) + r'\.(\w+)'
        replacement = f'{alias_name}.\\1'
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result


# ==================== A：normalized  token -> SQL ====================

_TOKEN_RE = re.compile(
    r"""
    (?:--[^\n]*\n?)|             # line comment
    (?:/\*.*?\*/)|               # block comment
    (?:'[^']*(?:''[^']*)*')|     # single-quoted string ('' escape)
    (?:"[^"]*(?:""[^"]*)*")|     # double-quoted string ("" escape)
    (?:\b\d+(?:\.\d+)?\b)|       # numbers
    (?:[A-Za-z_][\w$]*(?:\.[A-Za-z_][\w$]*)*)|  # identifiers with dots
    (?:<=|>=|<>|!=|==)|          # multi-char ops
    [(),;.*=<>+\-\/]             # single-char punct/op
    """,
    re.VERBOSE | re.DOTALL
)

def tokenize_sql_with_spans(sql: str) -> List[Dict[str, Any]]:
    """ SQL  token（ start/end ）， diff。"""
    toks: List[Dict[str, Any]] = []
    for m in _TOKEN_RE.finditer(sql):
        tok = m.group(0)
        if tok.startswith("--") or tok.startswith("/*"):
            continue
        toks.append({"tok": tok, "start": m.start(), "end": m.end()})
    return toks

def canon_token(tok: str) -> str:
    """token ：/，。"""
    if (tok.startswith("'") and tok.endswith("'")) or (tok.startswith('"') and tok.endswith('"')):
        return tok
    return tok.lower()

def first_divergence_token(norm_pred: str, norm_label: str) -> Dict[str, Any]:
    """
     normalized  token。
    ：index, pred_tok(dict), label_tok(dict), reason
    """
    a = tokenize_sql_with_spans(norm_pred)
    b = tokenize_sql_with_spans(norm_label)
    n = min(len(a), len(b))
    for i in range(n):
        if canon_token(a[i]["tok"]) != canon_token(b[i]["tok"]):
            return {"index": i, "pred": a[i], "label": b[i], "reason": "token_mismatch"}
    if len(a) != len(b):
        return {
            "index": n,
            "pred": a[n] if n < len(a) else None,
            "label": b[n] if n < len(b) else None,
            "reason": "length_mismatch",
        }
    return {"index": None, "pred": None, "label": None, "reason": "identical"}

def build_norm_to_orig_char_map(original: str, normalized: str) -> List[int]:
    """
     normalized  -> original （ len(normalized)+1）。
     SequenceMatcher ， normalize  token 。
    """
    sm = difflib.SequenceMatcher(a=original, b=normalized, autojunk=False)
    mapping = [0] * (len(normalized) + 1)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(j2 - j1):
                mapping[j1 + k] = i1 + k
        elif tag in ("replace", "insert"):
            for k in range(j2 - j1):
                mapping[j1 + k] = i1
        elif tag == "delete":
            pass

    mapping[len(normalized)] = len(original)

    for j in range(1, len(mapping)):
        if mapping[j] < mapping[j - 1]:
            mapping[j] = mapping[j - 1]
    return mapping

def whitespace_token_spans(s: str) -> List[Dict[str, Any]]:
    """ token（\\S+）， start/end。"""
    spans: List[Dict[str, Any]] = []
    for m in re.finditer(r"\S+", s):
        spans.append({"tok": m.group(0), "start": m.start(), "end": m.end()})
    return spans

def prefix_by_orig_div_char(original_pred: str, orig_div_char: int) -> Tuple[str, int]:
    """
     predict “”， \\S+ token ，
     prefix （ token ） prefix token 。
    """
    spans = whitespace_token_spans(original_pred)
    if not spans:
        return "", 0

    div_tok_idx = len(spans)
    for i, sp in enumerate(spans):
        if sp["start"] <= orig_div_char < sp["end"]:
            div_tok_idx = i
            break
        if orig_div_char < sp["start"]:
            div_tok_idx = i
            break

    prefix_tokens = [sp["tok"] for sp in spans[:div_tok_idx]]
    return " ".join(prefix_tokens), div_tok_idx


# ====================  step （： token_id ） ====================

def find_divergence_step_by_token_ids(
    sample_logits: List[Dict[str, Any]],
    prefix_str_for_ids: str,
    tokenizer: PreTrainedTokenizer,
) -> int:
    """
     tokenizer decode  prefix  step。

    ：
      1.  tokenizer.encode()  prefix， re-encode  generate()
          token ids （BPE tokenizer  Ġ ）。
      2.  decode gen_ids  prefix （endswith），
         （ Qwen with thinking mode） generate  <think>
         ， predict  batch_predict  clean 。
          prefix  decode ""。
      3.  return， last_match_k ，
          special token（ <｜end▁of▁sentence｜>） skip_special_tokens=True
         ， k  decode ， step。
    """
    prefix_str_for_ids = prefix_str_for_ids or ""

    gen_ids = [s.get("generated_token_id") for s in sample_logits if "generated_token_id" in s]
    if not gen_ids:
        return 0

    p_strip = prefix_str_for_ids.strip()
    if not p_strip:
        return 0

    p_normalized = " ".join(p_strip.split())

    #  decode gen_ids， decoded  prefix  k
    last_match_k = -1
    for k in range(1, len(gen_ids) + 1):
        decoded = tokenizer.decode(gen_ids[:k], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        d_strip = decoded.strip()
        d_normalized = " ".join(d_strip.split())

        # ：decoded  prefix 
        if d_normalized == p_normalized or d_normalized.endswith(p_normalized):
            last_match_k = k
            continue

        # prefix  decoded ： prefix ，
        if d_normalized.startswith(p_normalized):
            break

        # prefix  decoded （ think ），
        if p_normalized not in d_normalized:
            continue

        # prefix  decoded ： tokenizer ，
        break

    if last_match_k != -1:
        return min(last_match_k, len(gen_ids) - 1)
    return len(gen_ids) - 1

# ==================== （ tokenizer  chat_template）====================

class ChatTemplate:
    """ Chat Template ， tokenizer  apply_chat_template。
    
     LLaMA-Factory  template ，。
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(self, instruction: str, user_input: str) -> str:
        """ prompt， tokenizer.apply_chat_template。"""
        messages = []
        if instruction:
            messages.append({"role": "system", "content": instruction})
        messages.append({"role": "user", "content": user_input})
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt


# ====================  Tokenizer  ====================

def load_model_and_tokenizer(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    use_fast_tokenizer: bool = True,
    attn_implementation: str = "flash_attention_2",
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """ tokenizer。"""
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=use_fast_tokenizer,
        trust_remote_code=True,
        padding_side="left",
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Add pad token: {tokenizer.pad_token}")

    print(f"Loading model from {model_path}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )
        print(f"Model loaded with {attn_implementation}")
    except Exception as e:  # noqa: BLE001
        print(f"Failed to load with {attn_implementation}: {e}")
        print("Falling back to default attention...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

    model.eval()
    print("Model loaded successfully!")
    return model, tokenizer


# ====================  logits ====================

def batch_predict(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    max_new_tokens: int = 1024,
    batch_size: int = 8,
    cutoff_len: int = 3200,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 0.7,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
) -> tuple[List[str], List[List[Dict[str, Any]]]]:
    """ logits（：GPU  topk +  decode）。"""
    all_predictions: List[str] = []
    all_logits_info: List[List[Dict[str, Any]]] = []

    eos_token_id = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    for batch_start in tqdm(range(0, len(prompts), batch_size), desc="Batch prediction"):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cutoff_len,
        ).to(model.device)

        prompt_lengths = inputs["input_ids"].size(-1)
        actual_batch_size = inputs["input_ids"].size(0)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_tokens = outputs.sequences  # (batch, prompt_len + num_steps)
        scores = outputs.scores               # tuple of (batch, vocab_size) tensors
        num_steps = len(scores)

        if num_steps == 0:
            #  token（）
            for _ in range(actual_batch_size):
                all_logits_info.append([])
            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            all_predictions.extend([p.strip() for p in decoded_preds])
            continue

        # =====  1： GPU  steps  scores =====
        # scores: tuple of (batch, vocab_size) -> stack to (num_steps, batch, vocab_size)
        all_scores = torch.stack(scores, dim=0)  # (num_steps, batch, vocab_size)

        # =====  2： GPU  softmax + topk  steps =====
        # Reshape to (num_steps * batch, vocab_size) for batch topk
        flat_scores = all_scores.view(-1, all_scores.size(-1))  # (num_steps * batch, vocab_size)
        flat_probs = torch.softmax(flat_scores, dim=-1)         # (num_steps * batch, vocab_size)
        flat_top5_probs, flat_top5_indices = torch.topk(flat_probs, k=5, dim=-1)  # (num_steps * batch, 5)

        #  CPU （ GPU->CPU ）
        flat_top5_indices_cpu = flat_top5_indices.cpu().numpy()  # (num_steps * batch, 5)
        flat_top5_probs_cpu = flat_top5_probs.cpu().numpy()      # (num_steps * batch, 5)

        # =====  3： convert_ids_to_tokens  token ids =====
        #  tokenizer.decode（CPU ）
        gen_token_ids = generated_tokens[:, prompt_lengths:prompt_lengths + num_steps]  # (batch, num_steps)
        gen_token_ids = gen_token_ids.transpose(0, 1)  # (num_steps, batch)

        #  token ids
        all_ids_flat: List[int] = []
        for step_idx in range(num_steps):
            for b_idx in range(actual_batch_size):
                all_ids_flat.append(gen_token_ids[step_idx, b_idx].item())
                all_ids_flat.extend(flat_top5_indices_cpu[step_idx * actual_batch_size + b_idx].tolist())

        #  convert_ids_to_tokens（ decode  10-100 ）
        all_tokens = tokenizer.convert_ids_to_tokens(all_ids_flat)
        # convert_ids_to_tokens  token （ Ġ ），
        # ， .replace('Ġ', ' ').strip()

        # =====  4： =====
        batch_logits_info: List[List[Dict[str, Any]]] = [[] for _ in range(actual_batch_size)]
        token_idx = 0
        for step_idx in range(num_steps):
            for b_idx in range(actual_batch_size):
                generated_token = all_tokens[token_idx].replace('Ġ', ' ').strip() if all_tokens[token_idx] else ""
                top5_tokens = [
                    all_tokens[token_idx + j].replace('Ġ', ' ').strip() if all_tokens[token_idx + j] else ""
                    for j in range(1, 6)
                ]
                top5_probs = flat_top5_probs_cpu[step_idx * actual_batch_size + b_idx].tolist()
                top5_token_ids = flat_top5_indices_cpu[step_idx * actual_batch_size + b_idx].tolist()
                gen_token_id = gen_token_ids[step_idx, b_idx].item()

                batch_logits_info[b_idx].append({
                    "step": step_idx,
                    "generated_token": generated_token,
                    "generated_token_id": gen_token_id,
                    "top5_tokens": top5_tokens,
                    "top5_token_ids": top5_token_ids,
                    "top5_probs": top5_probs,
                })
                token_idx += 6

        all_logits_info.extend(batch_logits_info)

        # decode predictions（ prompt ）
        generated_tokens[:, :prompt_lengths] = pad_token_id
        decoded_preds = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        all_predictions.extend([pred.strip() for pred in decoded_preds])

    return all_predictions, all_logits_info


# ==================== （A + token_id  step） ====================

def extract_divergence_info_simple(
    dataset: List[Dict[str, Any]],
    predictions: List[str],
    logits_info: List[List[Dict[str, Any]]],
    tokenizer: PreTrainedTokenizer,
) -> tuple[
    List[List[Dict[str, Any]]],           # divergence_logits_all
    List[str],                             # prefixes
    List[str],                             # suffixes
    List[Optional[Dict[str, Any]]],        # semantic_infos ()
    List[str],                             # normalized_predicts
    List[str],                             # normalized_labels
    List[Dict[str, str]],                  # all_predict_aliases
    List[Dict[str, str]],                  # all_label_aliases
    List[Optional[Dict[str, Any]]],        # divergence_meta
]:
    """（A：normalized  token  SQL）。"""
    divergence_logits_all: List[List[Dict[str, Any]]] = []
    prefixes: List[str] = []
    suffixes: List[str] = []
    semantic_infos: List[Optional[Dict[str, Any]]] = []
    normalized_predicts: List[str] = []
    normalized_labels: List[str] = []
    all_predict_aliases: List[Dict[str, str]] = []
    all_label_aliases: List[Dict[str, str]] = []
    divergence_meta: List[Optional[Dict[str, Any]]] = []

    for i in range(len(dataset)):
        sample = dataset[i]
        label = sample.get("output", "") or ""
        predict = predictions[i] or ""
        sample_logits = logits_info[i]

        normalized_label = normalize_sql_remove_aliases(label)
        normalized_predict = normalize_sql_remove_aliases(predict)

        predict_aliases = extract_aliases_from_sql(predict)
        label_aliases = extract_aliases_from_sql(label)
        all_predict_aliases.append(predict_aliases)
        all_label_aliases.append(label_aliases)

        normalized_labels.append(normalized_label)
        normalized_predicts.append(normalized_predict)

        semantic_infos.append(None)

        # ========== A： normalized  token， predict ==========
        div = first_divergence_token(normalized_predict, normalized_label)

        # prefix_for_ids： token_id  step（， " ".join(...)）
        prefix_for_ids = ""

        if div["reason"] == "identical" or div["pred"] is None:
            prefix = predict.strip()
            suffix = ""
            prefixes.append(prefix)
            suffixes.append(suffix)
            divergence_meta.append(None)
            prefix_for_ids = prefix
        else:
            norm_div_start = int(div["pred"]["start"])

            norm2orig = build_norm_to_orig_char_map(predict, normalized_predict)
            orig_div_char = norm2orig[norm_div_start] if norm_div_start < len(norm2orig) else len(predict)

            prefix, prefix_tok_count = prefix_by_orig_div_char(predict, orig_div_char)

            pred_ws_spans = whitespace_token_spans(predict)

            # “ predict ” prefix_for_ids： token 
            if prefix_tok_count < len(pred_ws_spans):
                prefix_end_char = pred_ws_spans[prefix_tok_count]["start"]
            else:
                prefix_end_char = len(predict)
            prefix_for_ids = predict[:prefix_end_char].rstrip()

            # suffix： whitespace token （）
            suffix_tokens = [sp["tok"] for sp in pred_ws_spans[prefix_tok_count:]]
            suffix = " ".join(suffix_tokens)

            prefixes.append(prefix)
            suffixes.append(suffix)

            orig_div_tok = pred_ws_spans[prefix_tok_count]["tok"] if prefix_tok_count < len(pred_ws_spans) else None
            divergence_meta.append({
                "norm_div_index": div["index"],
                "norm_pred_token": div["pred"]["tok"],
                "norm_label_token": div["label"]["tok"] if div["label"] else None,
                "norm_pred_char": norm_div_start,
                "orig_pred_char": orig_div_char,
                "orig_pred_token": orig_div_tok,
                "orig_pred_token_index": prefix_tok_count,
                "reason": div["reason"],
            })

        # ==========  prefix_for_ids（token_id ） step ==========
        final_div_step = -1
        if sample_logits:
            final_div_step = find_divergence_step_by_token_ids(
                sample_logits=sample_logits,
                prefix_str_for_ids=prefix_for_ids,
                tokenizer=tokenizer,
            )

        if 0 <= final_div_step < len(sample_logits):
            divergence_logits_all.append([sample_logits[final_div_step]])
        else:
            divergence_logits_all.append([])

    return (
        divergence_logits_all,
        prefixes,
        suffixes,
        semantic_infos,
        normalized_predicts,
        normalized_labels,
        all_predict_aliases,
        all_label_aliases,
        divergence_meta,
    )


# ====================  ====================

def save_predictions(
    dataset: List[Dict[str, Any]],
    predictions: List[str],
    prompts: List[str],
    divergence_logits: List[List[Dict[str, Any]]],
    prefixes: List[str],
    suffixes: List[str],
    semantic_infos: List[Optional[Dict[str, Any]]],
    normalized_predicts: List[str],
    normalized_labels: List[str],
    predict_aliases: List[Dict[str, str]],
    label_aliases: List[Dict[str, str]],
    divergence_meta: List[Optional[Dict[str, Any]]],
    output_dir: str,
    output_filename: str = "generated_predictions_with_divergence.jsonl",
) -> str:
    """。"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, (sample, pred, prompt) in enumerate(zip(dataset, predictions, prompts)):
            item: Dict[str, Any] = {
                "prompt": prompt,
                "predict": pred,
                "label": sample.get("output", ""),
                "normalized_predict": normalized_predicts[i],
                "normalized_label": normalized_labels[i],
                "predict_aliases": predict_aliases[i],
                "label_aliases": label_aliases[i],
                "prefix": prefixes[i],
                "suffix": suffixes[i],
                "semantic_divergence": semantic_infos[i],
                "divergence_meta": divergence_meta[i] if i < len(divergence_meta) else None,
            }
            item["logits_top5"] = divergence_logits[i] if i < len(divergence_logits) else []
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Results saved to {output_path}")
    return output_path


# ====================  ====================

def main() -> None:
    parser = argparse.ArgumentParser(description=" Logit（A：normalized SQL + token_id step ）")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="spider_dev")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cutoff_len", type=int, default=3200)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--save_every_n_batches", type=int, default=20,
                        help="Save partial results every N batches (overwrite)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "checkpoint_raw.pt")
    output_path = os.path.join(args.output_dir, "generated_predictions_with_divergence.jsonl")

    dataset = load_spider_dataset(args.dataset_dir, f"{args.dataset}.json")
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)

    template = ChatTemplate(tokenizer)
    prompts = [template.apply_chat_template(s.get("instruction", ""), s.get("input", "")) for s in dataset]

    batch_size = args.per_device_eval_batch_size
    total_samples = len(prompts)
    total_batches = (total_samples + batch_size - 1) // batch_size

    #  checkpoint
    start_batch_idx = 0
    all_predictions: List[str] = []
    all_logits_info: List[List[Dict[str, Any]]] = []
    all_div_logits: List[List[Dict[str, Any]]] = []
    all_prefixes: List[str] = []
    all_suffixes: List[str] = []
    all_semantic: List[Optional[Dict[str, Any]]] = []
    all_norm_preds: List[str] = []
    all_norm_labels: List[str] = []
    all_pred_aliases: List[Dict[str, str]] = []
    all_label_aliases: List[Dict[str, str]] = []
    all_div_meta: List[Optional[Dict[str, Any]]] = []

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path} ...")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        all_predictions = ckpt["predictions"]
        all_logits_info = ckpt["logits_info"]
        all_div_logits = ckpt["div_logits"]
        all_prefixes = ckpt["prefixes"]
        all_suffixes = ckpt["suffixes"]
        all_semantic = ckpt["semantic"]
        all_norm_preds = ckpt["norm_preds"]
        all_norm_labels = ckpt["norm_labels"]
        all_pred_aliases = ckpt["pred_aliases"]
        all_label_aliases = ckpt["label_aliases"]
        all_div_meta = ckpt["div_meta"]
        start_batch_idx = ckpt["next_batch_idx"]
        print(f"  Resumed: {len(all_predictions)}/{total_samples} samples done, starting from batch {start_batch_idx}/{total_batches}")

    for batch_idx in range(start_batch_idx, total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_samples)
        batch_prompts = prompts[batch_start:batch_end]
        batch_dataset = dataset[batch_start:batch_end]

        batch_preds, batch_logits = batch_predict(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            max_new_tokens=args.max_new_tokens,
            batch_size=len(batch_prompts),
            cutoff_len=args.cutoff_len,
        )

        #  DeepSeek-R1  <think>...</think> 
        #  predict  label（ SQL）
        clean_preds = []
        for p in batch_preds:
            if "</think>" in p:
                p = p.split("</think>", 1)[1].strip()
            clean_preds.append(p)
        batch_preds = clean_preds

        #  batch  divergence extraction
        (
            batch_div_logits,
            batch_prefixes,
            batch_suffixes,
            batch_semantic,
            batch_norm_preds,
            batch_norm_labels,
            batch_pred_aliases,
            batch_label_aliases,
            batch_div_meta,
        ) = extract_divergence_info_simple(batch_dataset, batch_preds, batch_logits, tokenizer)

        # 
        all_predictions.extend(batch_preds)
        all_logits_info.extend(batch_logits)
        all_div_logits.extend(batch_div_logits)
        all_prefixes.extend(batch_prefixes)
        all_suffixes.extend(batch_suffixes)
        all_semantic.extend(batch_semantic)
        all_norm_preds.extend(batch_norm_preds)
        all_norm_labels.extend(batch_norm_labels)
        all_pred_aliases.extend(batch_pred_aliases)
        all_label_aliases.extend(batch_label_aliases)
        all_div_meta.extend(batch_div_meta)

        #  N  batch  checkpoint +  partial 
        next_batch_idx = batch_idx + 1
        if next_batch_idx % args.save_every_n_batches == 0 or batch_end == total_samples:
            print(f"[Batch {next_batch_idx}/{total_batches}] Saving checkpoint and partial results ({len(all_predictions)} samples)...")

            #  raw checkpoint（）
            torch.save({
                "predictions": all_predictions,
                "logits_info": all_logits_info,
                "div_logits": all_div_logits,
                "prefixes": all_prefixes,
                "suffixes": all_suffixes,
                "semantic": all_semantic,
                "norm_preds": all_norm_preds,
                "norm_labels": all_norm_labels,
                "pred_aliases": all_pred_aliases,
                "label_aliases": all_label_aliases,
                "div_meta": all_div_meta,
                "next_batch_idx": next_batch_idx,
            }, checkpoint_path)

            #  partial 
            save_predictions(
                dataset=dataset[:batch_end],
                predictions=all_predictions,
                prompts=prompts[:batch_end],
                divergence_logits=all_div_logits,
                prefixes=all_prefixes,
                suffixes=all_suffixes,
                semantic_infos=all_semantic,
                normalized_predicts=all_norm_preds,
                normalized_labels=all_norm_labels,
                predict_aliases=all_pred_aliases,
                label_aliases=all_label_aliases,
                divergence_meta=all_div_meta,
                output_dir=args.output_dir,
                output_filename="generated_predictions_with_divergence.jsonl",
            )

    #  checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Removed checkpoint: {checkpoint_path}")

    print(f"Inference completed! Total samples: {len(all_predictions)}")
    print(f"Final output: {output_path}")


if __name__ == "__main__":
    main()
