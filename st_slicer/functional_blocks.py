# st_slicer/functional_blocks.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Iterable, Dict, Optional
from .slicer import backward_slice   # 如果已经在上面 import 过就不用重复
from .criteria import SlicingCriterion
from .ast.nodes import Stmt, VarRef, ArrayAccess, FieldAccess, Literal, BinOp, CallExpr

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
    criteria: List[SlicingCriterion] = field(default_factory=list)
    node_ids: Set[int] = field(default_factory=set)
    stmts: List[Stmt] = field(default_factory=list)
    line_numbers: List[int] = field(default_factory=list)



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

def close_with_control_structures(
    stmt_set: Set[Stmt],
    parent_map: Dict[Stmt, Optional[Stmt]],
) -> Set[Stmt]:
    """
    在原有语句集合基础上，补全所有必要的控制语句（IfStmt / ForStmt 的外层骨架）。

    做法：
      - 对集合里的每个语句，沿 parent_map 一路向上找；
      - 遇到 IfStmt 或 ForStmt，就加入 closed 集合；
      - 继续往上，直到 None。
    """
    closed: Set[Stmt] = set(stmt_set)
    worklist: List[Stmt] = list(stmt_set)

    while worklist:
        st = worklist.pop()
        p = parent_map.get(st)
        while p is not None:
            if isinstance(p, (IfStmt, ForStmt)) and p not in closed:
                closed.add(p)
                worklist.append(p)
            p = parent_map.get(p)

    return closed


def nodes_to_sorted_ast_stmts(
    cluster_nodes: Set[int],
    ir2ast_stmt: List[Stmt],
    parent_map: Dict[Stmt, Optional[Stmt]],
) -> List[Stmt]:
    """
    从节点集合映射到 AST 语句集合，并按源码行号排序。
    会自动补全必要的 IfStmt / ForStmt 之类控制结构。
    """
    stmt_set: Set[Stmt] = set()
    for nid in cluster_nodes:
        if 0 <= nid < len(ir2ast_stmt):
            ast_stmt = ir2ast_stmt[nid]
            if ast_stmt is not None:
                stmt_set.add(ast_stmt)

    # 结构闭包：补全所需的控制结构骨架
    stmt_set = close_with_control_structures(stmt_set, parent_map)

    stmts = sorted(
        stmt_set,
        key=lambda s: (getattr(s.loc, "line", 0), getattr(s.loc, "column", 0)),
    )
    return stmts


def build_parent_map_from_ir2ast(ir2ast_stmt: List[Stmt]) -> Dict[Stmt, Optional[Stmt]]:
    """
    基于 ir2ast_stmt 粗略构造一个 parent_map: child_stmt -> parent_stmt (IfStmt / ForStmt / None)。

    思路：
      - ir2ast_stmt 里有本 POU 中几乎所有可能出现在切片里的 Stmt；
      - 对每个 Stmt，如果它是 IfStmt / ForStmt，就遍历它的 body，把 body 里的 Stmt 标记 parent；
      - 再递归处理嵌套结构。
    """
    uniq_stmts: Set[Stmt] = {st for st in ir2ast_stmt if st is not None}
    parent: Dict[Stmt, Optional[Stmt]] = {}

    def visit(stmt: Stmt, parent_stmt: Optional[Stmt]) -> None:
        # 只关心出现在 uniq_stmts 里的语句
        if stmt not in uniq_stmts:
            return
        # 第一次访问时记录 parent；后续不要覆盖
        if stmt not in parent:
            parent[stmt] = parent_stmt

        # 向下递归处理控制结构
        if isinstance(stmt, IfStmt):
            for child in stmt.then_body:
                visit(child, stmt)
            for _cond, body in stmt.elif_branches:
                for child in body:
                    visit(child, stmt)
            for child in stmt.else_body:
                visit(child, stmt)

        elif isinstance(stmt, ForStmt):
            for child in stmt.body:
                visit(child, stmt)

    # 把所有语句都当作「潜在根」跑一遍 visit，真正的 parent 只会在子访问时被设置
    for st in uniq_stmts:
        visit(st, None)

    return parent



def _scan_if_header_end(line_start: int, code_lines: List[str]) -> int:
    """
    从 IF 语句起始行向下扫描，直到遇到包含 THEN 的行，返回该行号。
    用于处理多行条件：
        IF cond1 OR
           cond2 OR
           cond3 THEN
    """
    n = len(code_lines)
    ln = line_start
    while ln <= n:
        text = code_lines[ln - 1].upper()
        if "THEN" in text:
            return ln
        ln += 1
    return line_start


def _scan_matching_end_if(line_start: int, code_lines: List[str]) -> int:
    """
    从 IF 语句起始行向下扫描，使用简单深度计数找到匹配的 END_IF 行号。
    支持嵌套 IF。
    """
    n = len(code_lines)
    depth = 0
    for ln in range(line_start, n + 1):
        text = code_lines[ln - 1].upper().strip()
        # 粗略判断 IF ... THEN（避免把注释等误算进去，可以再按需要收紧条件）
        if "IF" in text and "THEN" in text:
            depth += 1
        if "END_IF" in text:
            depth -= 1
            if depth == 0:
                return ln
    return line_start


def stmts_to_line_numbers(stmts: List[Stmt], code_lines: List[str]) -> List[int]:
    """
    把语句集合映射为源码行号集合，并按行号排序。
    针对 IfStmt，会额外加入：
        - 多行条件头部所有行（直到 THEN）
        - 匹配的 END_IF 行
    """
    sliced_lines: Set[int] = set()
    n = len(code_lines)

    for st in stmts:
        line_no = getattr(st.loc, "line", None)
        if line_no is None or not (1 <= line_no <= n):
            continue

        if isinstance(st, IfStmt):
            # 1) IF / ELSIF 头部（支持多行条件）
            header_end = _scan_if_header_end(line_no, code_lines)
            for ln in range(line_no, header_end + 1):
                sliced_lines.add(ln)

            # 2) 匹配 END_IF 行
            end_if_ln = _scan_matching_end_if(line_no, code_lines)
            if 1 <= end_if_ln <= n:
                sliced_lines.add(end_if_ln)

        else:
            # 普通语句，使用自身所在行
            sliced_lines.add(line_no)

    return sorted(sliced_lines)

# -------------------------------------------------
# 从 parent_block 中按行号子集构造子块
# -------------------------------------------------
def _build_block_from_lines(
    parent_block: FunctionalBlock,
    seg_lines: List[int],
    ir2ast_stmt: List[Stmt],
    code_lines: List[str],
    parent_map: Dict[Stmt, Optional[Stmt]],
) -> FunctionalBlock:
    """
    给定父功能块 + 一段连续/非连续行号，构造一个新的子功能块：
      - 从父块的 node_ids 中筛出落在这些行里的节点；
      - 基于这些节点重新生成 stmts / line_numbers。
    """
    seg_line_set = set(seg_lines)

    # 1) 找出属于这些行号的 node_ids
    sub_node_ids: Set[int] = set()
    for nid in parent_block.node_ids:
        if 0 <= nid < len(ir2ast_stmt):
            st = ir2ast_stmt[nid]
            if st is None:
                continue
            ln = getattr(st.loc, "line", None)
            if ln is not None and ln in seg_line_set:
                sub_node_ids.add(nid)

    # 2) 由子 node_ids 重新生成 stmts（带控制结构闭包）
    sub_stmts = nodes_to_sorted_ast_stmts(sub_node_ids, ir2ast_stmt, parent_map)

    # 3) 再由 stmts 生成精确的行号集合
    sub_lines = stmts_to_line_numbers(sub_stmts, code_lines)

    return FunctionalBlock(
        criteria=list(parent_block.criteria),
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
    parent_map: Dict[Stmt, Optional[Stmt]],
) -> List[FunctionalBlock]:
    """
    如果一个功能块太大，就按行号大致切分成若干子块。
    """
    lines = sorted(block.line_numbers)
    if len(lines) <= max_lines:
        return [block]

    segments: List[List[int]] = []
    current: List[int] = []

    for ln in lines:
        if not current:
            current = [ln]
            continue
        # 按简单策略分段：连续行放一起，超过 max_lines 则开启新段
        if ln == current[-1] + 1 and len(current) < max_lines:
            current.append(ln)
        else:
            segments.append(current)
            current = [ln]
    if current:
        segments.append(current)

    sub_blocks: List[FunctionalBlock] = []
    for seg in segments:
        if len(seg) < min_lines:
            # 太小的片段可以选择丢弃，或者和邻近片段合并；
            # 这里简单丢弃，保持行为和原来一致即可
            continue
        b = _build_block_from_lines(block, seg, ir2ast_stmt, code_lines, parent_map)
        sub_blocks.append(b)

    # 如果全被丢弃了，至少保留原 block，避免返回空列表
    return sub_blocks or [block]


# -------------------------------------------------
# 对所有块做大小规范化
# -------------------------------------------------
def normalize_block_sizes(
    blocks: List[FunctionalBlock],
    ir2ast_stmt: List[Stmt],
    code_lines: List[str],
    min_lines: int,
    max_lines: int,
    parent_map: Dict[Stmt, Optional[Stmt]],
) -> List[FunctionalBlock]:
    """
    对功能块大小做规范化：
      - 太大的块拆分；
      - 太小的块可以选择合并或丢弃（当前实现只做简单拆分）。
    """
    normalized: List[FunctionalBlock] = []

    for block in blocks:
        line_count = len(block.line_numbers)
        if line_count > max_lines:
            sub_blocks = _split_block_by_size(
                block, ir2ast_stmt, code_lines, min_lines, max_lines, parent_map
            )
            normalized.extend(sub_blocks)
        else:
            normalized.append(block)

    return normalized



# --------（可选）变量收集，后面用于构造小 PROGRAM --------

def collect_vars_in_expr(
    expr: Expr,
    vars_used: Set[str],
    funcs_used: Set[str] | None = None,
) -> None:
    if expr is None:
        return

    # 1) 变量引用
    if isinstance(expr, VarRef):
        vars_used.add(expr.name)

    # 2) 数组访问：递归 base 和 index
    elif isinstance(expr, ArrayAccess):
        collect_vars_in_expr(expr.base, vars_used, funcs_used)
        collect_vars_in_expr(expr.index, vars_used, funcs_used)

    # 3) 结构体字段访问：递归 base
    elif isinstance(expr, FieldAccess):
        collect_vars_in_expr(expr.base, vars_used, funcs_used)

    # 4) 二元运算：递归左右
    elif isinstance(expr, BinOp):
        collect_vars_in_expr(expr.left, vars_used, funcs_used)
        collect_vars_in_expr(expr.right, vars_used, funcs_used)

    # 5) 函数/FB 调用表达式：只递归参数，不把 func 算变量
    elif isinstance(expr, CallExpr):
        # 如需统计函数名，可以写入 funcs_used
        if funcs_used is not None:
            funcs_used.add(expr.func)
        for arg in expr.args:
            collect_vars_in_expr(arg, vars_used, funcs_used)

    # 6) 字面量：忽略
    elif isinstance(expr, Literal):
        return

    else:
        # 以后有新的 Expr 子类，再在这里补分支即可
        return


def collect_vars_in_stmt(stmt: Stmt,
                         vars_used: Set[str],
                         funcs_used: Set[str] | None = None):
    if isinstance(stmt, Assignment):
        collect_vars_in_expr(stmt.target, vars_used, funcs_used)
        collect_vars_in_expr(stmt.value, vars_used, funcs_used)

    elif isinstance(stmt, IfStmt):
        collect_vars_in_expr(stmt.cond, vars_used, funcs_used)
        for s in stmt.then_body:
            collect_vars_in_stmt(s, vars_used, funcs_used)
        for cond, body in stmt.elif_branches:
            collect_vars_in_expr(cond, vars_used, funcs_used)
            for s in body:
                collect_vars_in_stmt(s, vars_used, funcs_used)
        for s in stmt.else_body:
            collect_vars_in_stmt(s, vars_used, funcs_used)

    elif isinstance(stmt, ForStmt):
        # 循环变量本身也要声明，所以放进 vars_used
        vars_used.add(stmt.var)
        collect_vars_in_expr(stmt.start, vars_used, funcs_used)
        collect_vars_in_expr(stmt.end, vars_used, funcs_used)
        if stmt.step is not None:
            collect_vars_in_expr(stmt.step, vars_used, funcs_used)
        for s in stmt.body:
            collect_vars_in_stmt(s, vars_used, funcs_used)

    elif isinstance(stmt, CallStmt):
        # 这里只遍历参数，不把 stmt.fb_name 加到 vars_used 里
        for arg in stmt.args:
            collect_vars_in_expr(arg, vars_used, funcs_used)



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
    
    # 0) 基于 ir2ast_stmt 构造 parent_map（只做一次）
    parent_map = build_parent_map_from_ir2ast(ir2ast_stmt)

    # 1) 做切片、聚类 ...
    all_slices: List[Tuple[SlicingCriterion, Set[int]]] = []
    for crit in criteria:
        nodes = compute_slice_nodes(prog_pdg, crit.node_id)
        all_slices.append((crit, nodes))

    clusters = cluster_slices(all_slices, overlap_threshold=overlap_threshold)

    blocks: List[FunctionalBlock] = []
    for cluster in clusters:
        node_ids = cluster["nodes"]
        crits = cluster["criteria"]

        stmts = nodes_to_sorted_ast_stmts(node_ids, ir2ast_stmt, parent_map)
        line_numbers = stmts_to_line_numbers(stmts, code_lines)

        block = FunctionalBlock()
        block.criteria = crits
        block.node_ids = set(node_ids)
        block.stmts = stmts
        block.line_numbers = line_numbers
        blocks.append(block)

    # 4) 大小规范化 ...
    blocks = normalize_block_sizes(
        blocks,
        ir2ast_stmt=ir2ast_stmt,
        code_lines=code_lines,
        min_lines=min_lines,
        max_lines=max_lines,
        parent_map=parent_map,
    )
    return blocks

