# st_slicer/functional_blocks.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Tuple

from .ast.nodes import (
    Expr,
    VarRef,
    ArrayAccess,
    FieldAccess,
    Literal,
    BinOp,
    Stmt,
    Assignment,
    IfStmt,
    ForStmt,
    CallStmt,
)
from .slicer import backward_slice
from .criteria import SlicingCriterion


@dataclass
class FunctionalBlock:
    """
    一个功能块的抽象：由若干切片准则 + 节点集合 + AST 语句 + 源码行号组成。
    后续你可以在这里加：vars_used, var_decls, block_program_ast 等字段。
    """
    criteria: List[SlicingCriterion]
    node_ids: Set[int]
    stmts: List[Stmt]
    line_numbers: List[int]


# -----------------------
# 低层工具函数
# -----------------------

def compute_slice_nodes(prog_pdg, start_node_id: int) -> Set[int]:
    """
    对给定起始节点做一次后向切片。
    如需按变量过滤，可在这里扩展；目前直接复用 backward_slice。
    """
    return backward_slice(prog_pdg, [start_node_id])


def cluster_slices(
    all_slices: List[Tuple[SlicingCriterion, Set[int]]],
    overlap_threshold: float = 0.5,
) -> List[dict]:
    """
    输入: all_slices = [(criterion, node_set), ...]
    输出: clusters = [
        {
            "nodes": set[int],                 # 该簇中所有节点的并集
            "criteria": [criterion, ...],      # 属于这个簇的所有准则
        },
        ...
    ]
    overlap_threshold: 两个切片的重叠比例 >= 此阈值时归为同一簇。
    """
    clusters: List[dict] = []

    for crit, node_set in all_slices:
        placed = False
        for cluster in clusters:
            cluster_nodes: Set[int] = cluster["nodes"]
            inter = len(cluster_nodes & node_set)
            denom = min(len(cluster_nodes), len(node_set))
            if denom == 0:
                continue
            overlap = inter / denom
            if overlap >= overlap_threshold:
                cluster["nodes"] |= node_set
                cluster["criteria"].append(crit)
                placed = True
                break

        if not placed:
            clusters.append(
                {
                    "nodes": set(node_set),
                    "criteria": [crit],
                }
            )

    return clusters


def nodes_to_sorted_ast_stmts(
    cluster_nodes: Set[int],
    ir2ast_stmt: List[Stmt],
) -> List[Stmt]:
    """
    从节点集合映射到 AST 语句集合，并按源码行号排序。
    """
    stmt_set: Set[Stmt] = set()
    for nid in cluster_nodes:
        if 0 <= nid < len(ir2ast_stmt):
            ast_stmt = ir2ast_stmt[nid]
            if ast_stmt is not None:
                stmt_set.add(ast_stmt)

    stmts = sorted(
        stmt_set,
        key=lambda s: (getattr(s.loc, "line", 0), getattr(s.loc, "column", 0)),
    )
    return stmts


def stmts_to_line_numbers(stmts: List[Stmt], code_lines: List[str]) -> List[int]:
    """
    把语句集合映射为源码行号集合，并按行号排序。
    """
    sliced_lines: Set[int] = set()
    for st in stmts:
        line_no = getattr(st.loc, "line", None)
        if line_no is not None and 1 <= line_no <= len(code_lines):
            sliced_lines.add(line_no)

    return sorted(sliced_lines)


# --------（可选）变量收集，后面用于构造小 PROGRAM --------

def collect_vars_in_expr(expr: Expr, acc: Set[str]):
    if isinstance(expr, VarRef):
        acc.add(expr.name)
    elif isinstance(expr, ArrayAccess):
        collect_vars_in_expr(expr.base, acc)
        collect_vars_in_expr(expr.index, acc)
    elif isinstance(expr, FieldAccess):
        collect_vars_in_expr(expr.base, acc)
    elif isinstance(expr, BinOp):
        collect_vars_in_expr(expr.left, acc)
        collect_vars_in_expr(expr.right, acc)
    # Literal 不含变量，略过


def collect_vars_in_stmt(stmt: Stmt, acc: Set[str]):
    if isinstance(stmt, Assignment):
        collect_vars_in_expr(stmt.target, acc)
        collect_vars_in_expr(stmt.value, acc)
    elif isinstance(stmt, IfStmt):
        collect_vars_in_expr(stmt.cond, acc)
        for s in stmt.then_body:
            collect_vars_in_stmt(s, acc)
        for cond, body in stmt.elif_branches:
            collect_vars_in_expr(cond, acc)
            for s in body:
                collect_vars_in_stmt(s, acc)
        for s in stmt.else_body:
            collect_vars_in_stmt(s, acc)
    elif isinstance(stmt, ForStmt):
        acc.add(stmt.var)
        collect_vars_in_expr(stmt.start, acc)
        collect_vars_in_expr(stmt.end, acc)
        if stmt.step is not None:
            collect_vars_in_expr(stmt.step, acc)
        for s in stmt.body:
            collect_vars_in_stmt(s, acc)
    elif isinstance(stmt, CallStmt):
        for arg in stmt.args:
            collect_vars_in_expr(arg, acc)


def collect_vars_in_block(stmts: List[Stmt]) -> Set[str]:
    vars_used: Set[str] = set()
    for s in stmts:
        collect_vars_in_stmt(s, vars_used)
    return vars_used


# -----------------------
# 高层封装：一键“功能块划分”
# -----------------------

def extract_functional_blocks(
    prog_pdg,
    criteria: List[SlicingCriterion],
    ir2ast_stmt: List[Stmt],
    code_lines: List[str],
    overlap_threshold: float = 0.5,
) -> List[FunctionalBlock]:
    """
    核心入口：给定 PDG + 准则 + IR→AST 映射 + 源码行，返回若干功能块。
    每个功能块是一个 FunctionalBlock 实例。
    """

    if not criteria:
        return []

    # 1) 对所有准则做细粒度切片
    all_slices: List[Tuple[SlicingCriterion, Set[int]]] = []
    for crit in criteria:
        nodes = compute_slice_nodes(prog_pdg, crit.node_id)
        all_slices.append((crit, nodes))

    # 2) 基于节点重叠度做聚类
    clusters = cluster_slices(all_slices, overlap_threshold=overlap_threshold)

    # 3) 每个簇映射成一个 FunctionalBlock
    blocks: List[FunctionalBlock] = []
    for cluster in clusters:
        node_ids: Set[int] = cluster["nodes"]
        stmts = nodes_to_sorted_ast_stmts(node_ids, ir2ast_stmt)
        if not stmts:
            continue
        line_numbers = stmts_to_line_numbers(stmts, code_lines)
        block = FunctionalBlock(
            criteria=list(cluster["criteria"]),
            node_ids=node_ids,
            stmts=stmts,
            line_numbers=line_numbers,
        )
        blocks.append(block)

    return blocks
