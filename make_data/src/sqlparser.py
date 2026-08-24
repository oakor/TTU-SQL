#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SQL 

 sqlglot  SQL  AST，
“”， SQL 。

：
- （ t1 vs a）， ID（T1, T2, ...）。
-  FROM  INNER JOIN / （）。
-  WHERE / HAVING  AND ，。
- （FROM -> WHERE -> GROUP BY -> HAVING -> SELECT -> ORDER BY -> LIMIT）
  ，“”。

：
-  SQL ， logits；
   first_divergence_stage 
   token 。
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import sqlglot
from sqlglot import exp


# ==================== SQL  ====================


def normalize_sql_string(sql: str) -> str:
    """ SQL 。

    ：
    - ；
    - （）；
    - 。
    """

    if sql is None:
        return ""

    # 
    sql = sql.replace("\r", " ").replace("\n", " ")
    # 
    sql = re.sub(r"\s+", " ", sql)
    # 
    return sql.strip()


# ====================  ====================


@dataclass
class QuerySignature:
    """， SQL 。

    、（list / tuple / str）。
    """

    tables: List[str]
    joins: List[str]
    filters: List[str]
    group_by: List[str]
    having: List[str]
    select: List[str]
    order_by: List[str]
    limit: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompareResult:
    """SQL 。"""

    gold_sql_norm: str
    pred_sql_norm: str
    gold_signature: QuerySignature
    pred_signature: QuerySignature
    is_equivalent: bool
    first_divergence_stage: Optional[str]
    first_divergence_text: Optional[str]  # ： predict 
    differences: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gold_sql_norm": self.gold_sql_norm,
            "pred_sql_norm": self.pred_sql_norm,
            "gold_signature": self.gold_signature.to_dict(),
            "pred_signature": self.pred_signature.to_dict(),
            "is_equivalent": self.is_equivalent,
            "first_divergence_stage": self.first_divergence_stage,
            "first_divergence_text": self.first_divergence_text,
            "differences": self.differences,
        }


# ==================== AST  ====================


def _parse_sql(sql: str, dialect: str = "sqlite") -> Optional[exp.Expression]:
    """ sqlglot  SQL， AST。"""

    sql = normalize_sql_string(sql)
    if not sql:
        return None

    try:
        return sqlglot.parse_one(sql, read=dialect)
    except Exception as exc:  # noqa: BLE001
        print(f"[sqlparser] parse error ({dialect}): {exc} \nSQL: {sql}")
        return None


def _build_table_alias_map(query: exp.Select) -> Dict[str, str]:
    """/ ID ， {"t1": "T1", "table": "T1"}。"""

    alias_map: Dict[str, str] = {}
    next_id = 1

    for table in query.find_all(exp.Table):
        name = table.alias_or_name
        if not name:
            continue
        if name not in alias_map:
            alias_map[name] = f"T{next_id}"
            next_id += 1

    return alias_map


def _normalize_column(col: exp.Column, alias_map: Dict[str, str]) -> str:
    table_name = col.table
    column_name = col.name

    if table_name and table_name in alias_map:
        return f"{alias_map[table_name]}.{column_name}"
    if table_name:
        return f"{table_name}.{column_name}"
    return column_name


def _expression_to_str(e: exp.Expression, alias_map: Dict[str, str]) -> str:
    """。"""

    if isinstance(e, exp.Column):
        return _normalize_column(e, alias_map)

    def _clone_and_normalize(node: exp.Expression) -> exp.Expression:
        node = node.copy()
        for col in node.find_all(exp.Column):
            norm = _normalize_column(col, alias_map)
            col.replace(exp.to_identifier(norm))
        return node

    return _clone_and_normalize(e).sql(dialect="sqlite")


# ====================  ====================


def _collect_tables_and_joins(
    query: exp.Select, alias_map: Dict[str, str]
) -> Tuple[List[str], List[str]]:
    """ FROM / JOIN 。"""

    tables_info: Dict[str, str] = {}
    join_repr: List[str] = []

    for table in query.find_all(exp.Table):
        alias_or_name = table.alias_or_name
        if not alias_or_name:
            continue
        norm_id = alias_map.get(alias_or_name)
        if not norm_id:
            continue
        real_name = table.this.name if isinstance(table.this, exp.Identifier) else alias_or_name
        tables_info[norm_id] = real_name

    # ：sqlglot  Join  .left ，
    #  JOIN 。“”，：
    #   - JOIN （INNER/LEFT/...）
    #   -  JOIN 
    #   - ON 
    #  + ， tables_info
    # 。
    for join in query.find_all(exp.Join):
        join_type = (join.kind or "INNER").upper()
        right = join.this

        def _table_norm(t: exp.Expression) -> str:
            if isinstance(t, exp.Table):
                alias_or_name = t.alias_or_name
                if alias_or_name and alias_or_name in alias_map:
                    return alias_map[alias_or_name]
                if alias_or_name:
                    return alias_or_name
            return t.sql(dialect="sqlite")

        right_s = _table_norm(right)

        on_expr = join.args.get("on")
        on_str = _expression_to_str(on_expr, alias_map) if on_expr is not None else ""

        if join_type == "INNER":
            # INNER JOIN ， JOIN 
            #  ON ，。
            join_repr.append(f"INNER({right_s}) ON {on_str}")
        else:
            #  LEFT/RIGHT/OUTER JOIN ，
            join_repr.append(f"{join_type}({right_s}) ON {on_str}")

    tables_list = [f"{tid}:{tables_info[tid]}" for tid in sorted(tables_info.keys())]
    join_repr.sort()

    return tables_list, join_repr


def _collect_conditions(
    expressions: List[exp.Expression], alias_map: Dict[str, str]
) -> List[str]:
    """ WHERE/HAVING ， AND +。"""

    parts: List[str] = []

    def _split_and(e: exp.Expression) -> List[exp.Expression]:
        if isinstance(e, exp.And):
            return _split_and(e.left) + _split_and(e.right)
        return [e]

    for expr_ in expressions:
        for cond in _split_and(expr_):
            parts.append(_expression_to_str(cond, alias_map))

    parts = sorted(set(parts))
    return parts


def _collect_group_by(query: exp.Select, alias_map: Dict[str, str]) -> List[str]:
    group = query.args.get("group")
    if not group:
        return []
    items = [_expression_to_str(e, alias_map) for e in group.expressions]
    return sorted(set(items))


def _collect_having(query: exp.Select, alias_map: Dict[str, str]) -> List[str]:
    having_expr = query.args.get("having")
    if not having_expr:
        return []
    return _collect_conditions([having_expr], alias_map)


def _collect_select(query: exp.Select, alias_map: Dict[str, str]) -> List[str]:
    cols: List[str] = []
    for proj in query.expressions:
        if isinstance(proj, exp.Alias):
            expr_str = _expression_to_str(proj.this, alias_map)
            alias_name = proj.alias
            if alias_name:
                cols.append(f"{expr_str} AS {alias_name}")
            else:
                cols.append(expr_str)
        else:
            cols.append(_expression_to_str(proj, alias_map))
    return cols


def _collect_order_by(query: exp.Select, alias_map: Dict[str, str]) -> List[str]:
    order = query.args.get("order")
    if not order:
        return []
    items: List[str] = []
    for o in order.expressions:
        this = o.this
        desc = bool(o.args.get("desc"))
        expr_str = _expression_to_str(this, alias_map)
        direction = "DESC" if desc else "ASC"
        items.append(f"{expr_str} {direction}")
    return items


def _collect_limit(query: exp.Select) -> Optional[int]:
    limit_expr = query.args.get("limit")
    if not limit_expr:
        return None
    try:
        value = limit_expr.args.get("expression") or limit_expr.this
        if isinstance(value, exp.Literal) and value.is_int:
            return int(value.name)
        return None
    except Exception:  # noqa: BLE001
        return None


def build_query_signature(sql: str, dialect: str = "sqlite") -> Optional[QuerySignature]:
    """ SQL 。"""

    query = _parse_sql(sql, dialect=dialect)
    if query is None:
        return None

    if not isinstance(query, exp.Select):
        first_select = next(query.find_all(exp.Select), None)
        if first_select is None:
            return None
        query = first_select

    alias_map = _build_table_alias_map(query)

    tables, joins = _collect_tables_and_joins(query, alias_map)

    where_expr = query.args.get("where")
    filters = _collect_conditions([where_expr.this], alias_map) if where_expr else []

    group_by = _collect_group_by(query, alias_map)
    having = _collect_having(query, alias_map)

    select_cols = _collect_select(query, alias_map)
    order_by = _collect_order_by(query, alias_map)
    limit_value = _collect_limit(query)

    return QuerySignature(
        tables=tables,
        joins=joins,
        filters=filters,
        group_by=group_by,
        having=having,
        select=select_cols,
        order_by=order_by,
        limit=limit_value,
    )


# ==================== SQL  ====================


def compare_sql(
    gold_sql: str,
    pred_sql: str,
    dialect: str = "sqlite",
) -> CompareResult:
    """ SQL，。"""

    gold_norm = normalize_sql_string(gold_sql)
    pred_norm = normalize_sql_string(pred_sql)

    gold_sig = build_query_signature(gold_norm, dialect=dialect)
    pred_sig = build_query_signature(pred_norm, dialect=dialect)

    if gold_sig is None or pred_sig is None:
        return CompareResult(
            gold_sql_norm=gold_norm,
            pred_sql_norm=pred_norm,
            gold_signature=gold_sig or QuerySignature([], [], [], [], [], [], [], None),
            pred_signature=pred_sig or QuerySignature([], [], [], [], [], [], [], None),
            is_equivalent=False,
            first_divergence_stage="parse_error",
            first_divergence_text=None,
            differences=[
                {
                    "stage": "parse_error",
                    "gold": None if gold_sig is None else gold_sig.to_dict(),
                    "pred": None if pred_sig is None else pred_sig.to_dict(),
                }
            ],
        )

    stages = [
        ("tables", gold_sig.tables, pred_sig.tables),
        ("joins", gold_sig.joins, pred_sig.joins),
        ("filters", gold_sig.filters, pred_sig.filters),
        ("group_by", gold_sig.group_by, pred_sig.group_by),
        ("having", gold_sig.having, pred_sig.having),
        ("select", gold_sig.select, pred_sig.select),
        ("order_by", gold_sig.order_by, pred_sig.order_by),
        ("limit", gold_sig.limit, pred_sig.limit),
    ]

    differences: List[Dict[str, Any]] = []
    first_stage: Optional[str] = None
    first_text: Optional[str] = None

    for stage_name, g_val, p_val in stages:
        if g_val != p_val:
            if first_stage is None:
                first_stage = stage_name
                #  pred  gold 
                if isinstance(p_val, list) and isinstance(g_val, list):
                    diff_elements = [x for x in p_val if x not in g_val]
                    if diff_elements:
                        first_text = diff_elements[0]
                    elif len(p_val) != len(g_val):
                        # （）， pred 
                        first_text = p_val[-1] if p_val else stage_name
                else:
                    first_text = str(p_val)

            differences.append({"stage": stage_name, "gold": g_val, "pred": p_val})

    is_equal = len(differences) == 0

    return CompareResult(
        gold_sql_norm=gold_norm,
        pred_sql_norm=pred_norm,
        gold_signature=gold_sig,
        pred_signature=pred_sig,
        is_equivalent=is_equal,
        first_divergence_stage=first_stage,
        first_divergence_text=first_text,
        differences=differences,
    )


# ====================  ====================


def _main_cli() -> None:
    parser = argparse.ArgumentParser(description="Compare two SQL queries using sqlglot")
    parser.add_argument("--gold", type=str, required=True, help="Gold SQL")
    parser.add_argument("--pred", type=str, required=True, help="Predicted SQL")
    parser.add_argument(
        "--dialect",
        type=str,
        default="sqlite",
        help="SQL dialect for parsing (default: sqlite)",
    )

    args = parser.parse_args()

    result = compare_sql(args.gold, args.pred, dialect=args.dialect)
    print("=== GOLD (normalized) ===")
    print(result.gold_sql_norm)
    print("\n=== PRED (normalized) ===")
    print(result.pred_sql_norm)
    print("\n=== IS EQUIVALENT ===")
    print(result.is_equivalent)
    print("\n=== FIRST DIVERGENCE STAGE ===")
    print(result.first_divergence_stage)
    print("\n=== FIRST DIVERGENCE TEXT ===")
    print(result.first_divergence_text)
    print("\n=== DIFFERENCES ===")
    for diff in result.differences:
        print(f"[stage={diff['stage']}]\n  gold: {diff['gold']}\n  pred: {diff['pred']}\n")


if __name__ == "__main__":
    _main_cli()
