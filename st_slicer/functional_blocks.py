# st_slicer/functional_blocks.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Tuple, Iterable
from .slicer import backward_slice   # 如果已经在上面 import 过就不用重复
from .criteria import SlicingCriterion
from .ast.nodes import Stmt

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

# -------------------------------------------------
# 从 parent_block 中按行号子集构造子块
# -------------------------------------------------
def _build_block_from_lines(
    parent_block: FunctionalBlock,
    lines_subset: List[int],
    ir2ast_stmt: List[Stmt],
    code_lines: List[str],
) -> FunctionalBlock:
    """
    从父 block 中，基于行号子集重新构造一个子 FunctionalBlock。
    只保留那些位于 lines_subset 中的节点和语句。
    """
    line_set = set(lines_subset)

    # 1) 过滤 node_ids：只保留其对应语句行号在子集内的节点
    sub_node_ids: Set[int] = set()
    for nid in parent_block.node_ids:
        if 0 <= nid < len(ir2ast_stmt):
            st = ir2ast_stmt[nid]
            if st is None:
                continue
            ln = getattr(st.loc, "line", None)
            if ln in line_set:
                sub_node_ids.add(nid)

    # 2) 根据新的 node_ids 重新得到 stmts & line_numbers
    sub_stmts = nodes_to_sorted_ast_stmts(sub_node_ids, ir2ast_stmt)
    sub_lines = stmts_to_line_numbers(sub_stmts, code_lines)

    return FunctionalBlock(
        criteria=list(parent_block.criteria),   # 先沿用父块的准则集合
        node_ids=sub_node_ids,
        stmts=sub_stmts,
        line_numbers=sub_lines,
    )

# -------------------------------------------------
# 拆分“过大”的块
# -------------------------------------------------
def _split_block_by_size(
    block: FunctionalBlock,
    ir2ast_stmt: List[Stmt],
    code_lines: List[str],
    min_lines: int,
    max_lines: int,
    max_line_gap: int = 5,
) -> List[FunctionalBlock]:
    """
    将一个过大的 block 按源码行号拆成若干子块，每个子块目标行数 <= max_lines。
    拆分方式：
      1. 先按“行号间断点”（相邻行差值 > max_line_gap）切成若干段；
      2. 对每段再按 max_lines 做均匀切分。
    """
    lines = sorted(block.line_numbers)
    if len(lines) <= max_lines:
        return [block]

    # Step 1: 按行号 gap 切割成若干连续段
    segments: List[List[int]] = []
    cur_seg: List[int] = []
    for ln in lines:
        if not cur_seg:
            cur_seg = [ln]
            continue
        if ln - cur_seg[-1] <= max_line_gap:
            cur_seg.append(ln)
        else:
            segments.append(cur_seg)
            cur_seg = [ln]
    if cur_seg:
        segments.append(cur_seg)

    # Step 2: 对每个 segment 再按 max_lines 切分
    sub_blocks: List[FunctionalBlock] = []
    for seg in segments:
        if len(seg) <= max_lines:
            b = _build_block_from_lines(block, seg, ir2ast_stmt, code_lines)
            if len(b.line_numbers) >= min_lines:
                sub_blocks.append(b)
        else:
            # 长 segment 再等分成若干 chunk
            seg_lines = seg
            start = 0
            while start < len(seg_lines):
                chunk = seg_lines[start : start + max_lines]
                b = _build_block_from_lines(block, chunk, ir2ast_stmt, code_lines)
                if len(b.line_numbers) >= min_lines:
                    sub_blocks.append(b)
                start += max_lines   # 也可以改成 max_lines//2 产生重叠窗口

    return sub_blocks

# -------------------------------------------------
# 对所有块做大小规范化
# -------------------------------------------------
def normalize_block_sizes(
    blocks: List[FunctionalBlock],
    ir2ast_stmt: List[Stmt],
    code_lines: List[str],
    min_lines: int = 20,
    max_lines: int = 150,
) -> List[FunctionalBlock]:
    """
    约束功能块大小，使其行数在 [min_lines, max_lines] 之间：
      - 丢弃太小的块（< min_lines），目前做法是直接丢弃；
      - 拆分太大的块（> max_lines）为多个子块。
    """
    normalized: List[FunctionalBlock] = []

    for block in blocks:
        n_lines = len(block.line_numbers)
        if n_lines < min_lines:
            # 暂时直接丢弃; 后续可以改成“就近合并”
            continue
        if n_lines > max_lines:
            sub_blocks = _split_block_by_size(
                block, ir2ast_stmt, code_lines, min_lines, max_lines
            )
            normalized.extend(sub_blocks)
        else:
            normalized.append(block)

    return normalized


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
    min_lines: int = 20,
    max_lines: int = 150,
) -> List[FunctionalBlock]:
    # 1) 对所有准则做细粒度切片
    all_slices: List[Tuple[SlicingCriterion, Set[int]]] = []
    for crit in criteria:
        nodes = compute_slice_nodes(prog_pdg, crit.node_id)
        all_slices.append((crit, nodes))

    # 2) 聚类
    clusters = cluster_slices(all_slices, overlap_threshold=overlap_threshold)

    # 3) 初始功能块（不控制大小）
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

    # 4) 大小规范化
    blocks = normalize_block_sizes(
        blocks,
        ir2ast_stmt=ir2ast_stmt,
        code_lines=code_lines,
        min_lines=min_lines,
        max_lines=max_lines,
    )

    return blocks

