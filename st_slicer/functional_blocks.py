# st_slicer/functional_blocks.py

from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from typing import  Any, List, Set, Tuple, Iterable, Dict, Optional
from .slicer import backward_slice   # 如果已经在上面 import 过就不用重复
from .criteria import SlicingCriterion
from .ast.nodes import Stmt, VarRef, ArrayAccess, FieldAccess, Literal, BinOp, CallExpr
import re

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

# --- 轻量级：去掉 ST 注释，便于识别关键字 ---
_BLOCK_COMMENT_RE = re.compile(r"\(\*.*?\*\)", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//.*?$", re.MULTILINE)

def _strip_st_comments(s: str) -> str:
    s = _BLOCK_COMMENT_RE.sub("", s)
    s = _LINE_COMMENT_RE.sub("", s)
    return s


def _norm_line(code_line: str) -> str:
    """去注释、去空白、统一大写，便于关键字识别。"""
    t = _strip_st_comments(code_line).strip()
    return t.upper()


def _is_if_start(t: str) -> bool:
    """
    判断 IF 头开始（排除 ELSIF）。
    兼容：IF ... THEN（THEN 可以在本行出现；多行 THEN 的情况由 _scan_if_header_end 处理）
    这里只要识别到以 IF 开头即可入栈，END/THEN 细节不在此处强依赖。
    """
    if not t.startswith("IF "):
        return False
    if t.startswith("ELSIF "):
        return False
    return True


def _is_elsif(t: str) -> bool:
    return t.startswith("ELSIF ")


def _is_else(t: str) -> bool:
    # ST 中 ELSE 通常是单词，后面可能有注释
    return t == "ELSE" or t.startswith("ELSE ")


def _is_end_if(t: str) -> bool:
    # 允许 END_IF; 或 END_IF
    return t.startswith("END_IF")


@dataclass
class _IfFrame:
    if_line: int                          # IF 起始行（IF 关键行）
    current_branch_head: int              # 当前分支头行（IF/ELSIF/ELSE 的行号）
    branches: List[tuple]                 # (branch_head, body_start, body_end)


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
    """
    uniq_stmts: Set[Stmt] = {st for st in ir2ast_stmt if st is not None}
    parent: Dict[Stmt, Optional[Stmt]] = {}

    def visit(stmt: Stmt, parent_stmt: Optional[Stmt]) -> None:
        # 只关心出现在 uniq_stmts 里的语句
        if stmt not in uniq_stmts:
            return

        # 允许从 None 升级到一个真实的 parent
        if stmt not in parent or (parent[stmt] is None and parent_stmt is not None):
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

    # 把所有语句都当作「潜在根」跑一遍 visit
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
    支持嵌套 IF，并且不会把 ELSIF 误算成新的 IF。
    """
    n = len(code_lines)
    depth = 0
    for ln in range(line_start, n + 1):
        text = code_lines[ln - 1].upper().strip()

        # 先处理 END_IF，防止负深度
        if "END_IF" in text:
            depth -= text.count("END_IF")
            if depth == 0:
                return ln

        # 去掉 ELSIF 再判断 IF ... THEN
        tmp = text.replace("ELSIF", "")
        if "IF" in tmp and "THEN" in tmp:
            depth += 1

    return line_start


def patch_if_structure(
    sliced_lines: Set[int],
    code_lines: List[str],
    *,
    ensure_end_if: bool = True,
) -> Set[int]:
    """
    在 sliced_lines 基础上做“结构补全”：
      - 如果某个 IF 的某个分支 body 内有任意行被切入，则把该分支头行（ELSIF/ELSE/IF）补进去
      - 可选：如果 IF 的任意 body 行被切入，则确保对应 END_IF 行也在 sliced_lines（防御性补全）

    注意：
      - 该函数不会把整个 IF body 拉进来，只补关键结构行，避免分块膨胀
      - 支持嵌套 IF（用栈）
      - 假设 code_lines 是 1-based 行号映射到 index=0 的 list
    """
    n = len(code_lines)
    patched: Set[int] = set(sliced_lines)

    stack: List[_IfFrame] = []

    for ln in range(1, n + 1):
        t = _norm_line(code_lines[ln - 1])
        if not t:
            continue

        # 1) 新 IF 入栈
        if _is_if_start(t) and not _is_elsif(t):
            stack.append(_IfFrame(
                if_line=ln,
                current_branch_head=ln,  # 第一个分支头就是 IF 行
                branches=[]
            ))
            continue

        # 2) 分支切换：ELSIF / ELSE
        if _is_elsif(t) or _is_else(t):
            if not stack:
                continue
            frame = stack[-1]
            # 结束上一个分支 body：从 “上一分支头行 + 1” 到 “当前分支头行 - 1”
            body_start = frame.current_branch_head + 1
            body_end = ln - 1
            frame.branches.append((frame.current_branch_head, body_start, body_end))

            # 当前行成为新分支头
            frame.current_branch_head = ln
            continue

        # 3) IF 结束：END_IF
        if _is_end_if(t):
            if not stack:
                continue
            frame = stack.pop()

            # 收尾最后一个分支 body
            body_start = frame.current_branch_head + 1
            body_end = ln - 1
            frame.branches.append((frame.current_branch_head, body_start, body_end))

            # 判断：此 IF 是否“被切片触及”
            if_touched = False

            # 对每个分支：如果 body 中有任意被切入行，则补该分支头行（IF/ELSIF/ELSE）
            for branch_head, b_start, b_end in frame.branches:
                if b_start <= b_end:
                    # 只要 body 范围内存在一行属于 sliced_lines
                    # （这里用 any + range 会慢；用集合交更快）
                    # 但 b_start..b_end 可能很大，所以用线性扫描的早停版本
                    touched_branch = False
                    for L in patched:
                        if b_start <= L <= b_end:
                            touched_branch = True
                            break
                    if touched_branch:
                        patched.add(branch_head)
                        if_touched = True

            # 若 IF 的任何分支 body 被触及，则可选补 END_IF
            if ensure_end_if and if_touched:
                patched.add(ln)

            continue

        # 其他行不处理

    # 如果文件末尾栈未清空，说明存在不闭合 IF（通常是源码问题或解析不完整）
    # 这里不强行补，避免引入错误行；需要的话你可以在此处记录 warning。

    return patched

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

    base_lines = stmts_to_line_numbers(sub_stmts, code_lines)
    fixed_lines = patch_if_structure(base_lines, code_lines, ensure_end_if=True)
    sub_lines = sorted(fixed_lines)

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
    约束：尽量不要在未闭合的 IF/END_IF 等控制结构中间切断。
    """
    lines = sorted(block.line_numbers)
    if len(lines) <= max_lines:
        return [block]

    segments: List[List[int]] = []
    current: List[int] = []
    ctrl_depth = 0  # 控制结构深度（IF/END_IF, FOR/END_FOR, CASE/END_CASE）

    def update_ctrl_depth(text: str, depth: int) -> int:
        u = text.upper()

        # 1) 先处理 END_*，防止负深度
        if "END_IF" in u:
            depth -= u.count("END_IF")
        if "END_FOR" in u:
            depth -= u.count("END_FOR")
        if "END_CASE" in u:
            depth -= u.count("END_CASE")

        # 2) 处理 IF / FOR / CASE 头部，排除 ELSIF
        tmp = u.replace("ELSIF", "")
        if " IF " in tmp or tmp.strip().startswith("IF(") or tmp.strip().startswith("IF "):
            depth += 1
        if " FOR " in u or u.strip().startswith("FOR "):
            depth += 1
        if " CASE " in u or u.strip().startswith("CASE "):
            depth += 1

        return depth

    for ln in lines:
        if not current:
            current = [ln]
        else:
            # 行号不连续，且当前段已达到最小长度，并且不在控制结构内部 → 可以切一刀
            if ln != current[-1] + 1 and len(current) >= min_lines and ctrl_depth == 0:
                segments.append(current)
                current = [ln]
            else:
                current.append(ln)

        # 更新控制深度（基于当前行）
        if 1 <= ln <= len(code_lines):
            ctrl_depth = update_ctrl_depth(code_lines[ln - 1], ctrl_depth)

        # 当前段超过 max_lines，且不在控制结构内部 → 强制切分
        if len(current) >= max_lines and ctrl_depth == 0:
            segments.append(current)
            current = []

    if current:
        segments.append(current)

    # 对太小的段做一次简单合并，避免出现大量极短小块
    merged_segments: List[List[int]] = []
    for seg in segments:
        if not merged_segments:
            merged_segments.append(seg)
            continue
        if len(seg) < min_lines:
            # 合并到前一个 segment
            merged_segments[-1].extend(seg)
        else:
            merged_segments.append(seg)

    sub_blocks: List[FunctionalBlock] = []
    for seg in merged_segments:
        seg = sorted(set(seg))
        if len(seg) < max(1, min_lines // 2):
            # 仍然太小的段可以选择丢弃
            continue
        b = _build_block_from_lines(block, seg, ir2ast_stmt, code_lines, parent_map)
        sub_blocks.append(b)

    # 如果全被丢弃了，至少保留原 block，避免返回空列表
    return sub_blocks or [block]



def split_blocks_by_stage(
    blocks: List[FunctionalBlock],
    ir2ast_stmt: List[Stmt],
    code_lines: List[str],
    parent_map: Dict[Stmt, Optional[Stmt]],
    min_lines: int = 5,
    stage_var_names: Tuple[str, ...] = ("stage", "Stage"),
) -> List[FunctionalBlock]:
    """
    在 normalize_block_sizes 之前调用：
      - 先按 Stage 切分一个 block，避免 0–4 状态混在一起；
      - 再交给 normalize_block_sizes 做按行数二次切分。

    如果某个 block 中根本没有 stage 相关标记，则原样返回该 block。
    """
    new_blocks: List[FunctionalBlock] = []

    for block in blocks:
        stage_sub_blocks = _split_block_by_stage(
            block,
            ir2ast_stmt=ir2ast_stmt,
            code_lines=code_lines,
            parent_map=parent_map,
            min_lines=min_lines,
            stage_var_names=stage_var_names,
        )
        new_blocks.extend(stage_sub_blocks)

    return new_blocks


def _split_block_by_stage(
    block: FunctionalBlock,
    ir2ast_stmt: List[Stmt],
    code_lines: List[str],
    parent_map: Dict[Stmt, Optional[Stmt]],
    min_lines: int,
    stage_var_names: Tuple[str, ...] = ("stage", "Stage"),
) -> List[FunctionalBlock]:
    """
    针对单个功能块做基于状态机的切分：
      - 利用 'IF stage = N' / 'stage := N' 作为切分点；
      - 在不同 Stage 之间形成多个子段；
      - 保留原有 block 的 stmts / node_ids 子集，通过 _build_block_from_lines 重建。
    """
    lines = sorted(block.line_numbers)
    if not lines:
        return [block]

    # 标记每一行是否是 stage 切分点
    def is_stage_marker(line_text: str) -> bool:
        lt = line_text.strip()
        if not lt:
            return False
        upper = lt.upper()

        # 统一把变量名也转成大写来匹配
        stage_patterns = []
        for name in stage_var_names:
            uname = name.upper()
            # IF stage = N THEN
            stage_patterns.append(rf"\bIF\b.*\b{uname}\b\s*=\s*[\w\.]+")
            # stage := N
            stage_patterns.append(rf"\b{uname}\b\s*:?=\s*[\w\.]+")

        for pat in stage_patterns:
            if re.search(pat, upper):
                return True
        return False

    segments: List[List[int]] = []
    current: List[int] = []
    saw_stage_marker = False

    for ln in lines:
        text = code_lines[ln - 1] if 0 < ln <= len(code_lines) else ""
        if is_stage_marker(text):
            saw_stage_marker = True
            # 如果当前已经积累了一些行，则先把当前段收尾
            if current:
                segments.append(current)
            # 从当前行起开启新段（确保标记行出现在某个 segment 里）
            current = [ln]
        else:
            # 普通行：累积到 current 段中
            if not current:
                current = [ln]
            else:
                current.append(ln)

    if current:
        segments.append(current)

    # 没有识别到任何 Stage 标记，直接返回原 block
    if not saw_stage_marker:
        return [block]

    # 针对 segments 做一个简单的“最小行数”处理：
    #   - 如果某段 < min_lines，则尝试和前一段合并；
    #   - 再不行就在最后兜底保留原 block。
    merged_segments: List[List[int]] = []
    for seg in segments:
        if not merged_segments:
            merged_segments.append(seg)
            continue
        if len(seg) < min_lines:
            # 合并到前一个 segment
            merged_segments[-1].extend(seg)
        else:
            merged_segments.append(seg)

    sub_blocks: List[FunctionalBlock] = []
    for seg in merged_segments:
        seg = sorted(set(seg))
        if len(seg) < max(1, min_lines // 2):
            # 太小的段，如果你希望继续丢弃也可以，这里默认还是保留一点
            continue
        b = _build_block_from_lines(block, seg, ir2ast_stmt, code_lines, parent_map)
        sub_blocks.append(b)

    return sub_blocks or [block]

def normalize_and_split_blocks(
    blocks: List[FunctionalBlock],
    ir2ast_stmt: List[Stmt],
    code_lines: List[str],
    min_lines: int,
    max_lines: int,
    parent_map: Dict[Stmt, Optional[Stmt]],
) -> List[FunctionalBlock]:
    """
    推荐的两级规范化：
      1. 基于状态机的语义切分（split_blocks_by_stage）
      2. 基于行数的尺寸规范化（normalize_block_sizes）
    """
    # 1) Stage-based semantic splitting
    stage_blocks = split_blocks_by_stage(
        blocks=blocks,
        ir2ast_stmt=ir2ast_stmt,
        code_lines=code_lines,
        parent_map=parent_map,
        min_lines=min_lines,
        stage_var_names=("stage", "Stage"),
    )

    # 2) Size-based splitting
    normalized_blocks = normalize_block_sizes(
        blocks=stage_blocks,
        ir2ast_stmt=ir2ast_stmt,
        code_lines=code_lines,
        min_lines=min_lines,
        max_lines=max_lines,
        parent_map=parent_map,
    )

    return normalized_blocks

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

def is_meaningful_block(block: FunctionalBlock, code_lines: list[str]) -> bool:
    lines = [code_lines[ln - 1].strip()
             for ln in sorted(block.line_numbers)
             if 1 <= ln <= len(code_lines)]

    # 1) 赋值语句数
    assign_count = sum(
        (":=" in t) and not t.upper().startswith("IF ") and not t.upper().startswith("ELSIF ")
        for t in lines
    )

    # 2) 控制结构平衡检查（简易）
    depth = 0
    for t in lines:
        u = t.upper()
        if "END_IF" in u:
            depth -= u.count("END_IF")
        tmp = u.replace("ELSIF", "")
        if "IF" in tmp and "THEN" in tmp:
            depth += 1
    balanced = (depth == 0)

    # 3) 最小行数
    min_len_ok = (len(lines) >= 10)

    return assign_count >= 3 and balanced and min_len_ok


# -------------------------------------------------
# 后处理 1：删除块中的“空 IF”
# -------------------------------------------------

def _remove_empty_ifs_in_block(block: FunctionalBlock, code_lines: List[str]) -> None:
    """
    在单个 FunctionalBlock 内删除“空 IF”结构：
        IF <cond> THEN
        END_IF;
    判定是基于该 block 的 line_numbers：
      - 只要在 IF 和 END_IF 之间，在本 block 中没有任何非空、非注释的行，
        则视为“空 IF”，从 block.line_numbers 中移除 IF 和 END_IF 这两行。

    注意：
      - 不修改 code_lines，只是让这个 block 不再引用这些行号。
      - 如果 IF 内有嵌套 IF 头出现在本 block 中，会被视为“有内容”，不会删除外层 IF。
    """
    if not block.line_numbers:
        return

    ln_set = set(block.line_numbers)
    to_remove: Set[int] = set()
    n = len(code_lines)

    for ln in sorted(block.line_numbers):
        if not (1 <= ln <= n):
            continue

        raw = code_lines[ln - 1]
        stripped = _strip_st_comments(raw).strip()
        upper = stripped.upper()

        # 只处理 IF 头（排除 ELSIF）
        if not _is_if_start(upper):
            continue

        end_ln = _scan_matching_end_if(ln, code_lines)
        if not (1 <= end_ln <= n):
            continue

        # 在 [ln+1, end_ln-1] 范围内、当前 block 的行号中，是否有“实质内容”
        has_content = False
        for inner_ln in ln_set:
            if ln < inner_ln < end_ln:
                inner_raw = code_lines[inner_ln - 1]
                inner_text = _strip_st_comments(inner_raw).strip()
                if inner_text != "":
                    # 任何非空文本都算内容（包括嵌套 IF/ELSIF/ELSE）
                    has_content = True
                    break

        if not has_content:
            # 视为“空 IF”，移除 IF 头和匹配的 END_IF 行（如果在本 block 中）
            to_remove.add(ln)
            if end_ln in ln_set:
                to_remove.add(end_ln)

    if to_remove:
        block.line_numbers = sorted(ln for ln in block.line_numbers if ln not in to_remove)


def remove_empty_ifs_in_blocks(
    blocks: List[FunctionalBlock],
    code_lines: List[str],
) -> List[FunctionalBlock]:
    """
    对所有功能块执行 _remove_empty_ifs_in_block。
    只修改每个 block 的 line_numbers，不改变 code_lines 本身。
    """
    for b in blocks:
        _remove_empty_ifs_in_block(b, code_lines)
    return blocks


# -------------------------------------------------
# 后处理 2：对完全重复的块去重（按源码文本）
# -------------------------------------------------

def dedup_blocks_by_code(
    blocks: List[FunctionalBlock],
    code_lines: List[str],
) -> List[FunctionalBlock]:
    """
    按“切出来的源码文本是否完全相同”对功能块去重：
      - 对每个 block，将其 line_numbers 对应的源码行（去掉行尾空白）拼成字符串作为 key；
      - key 完全相同的 block 只保留出现的第一份，其余丢弃。

    这可以去掉类似 BLOCK 1 / BLOCK 5 这种 PROGRAM 主体完全相同、
    只是名字不同的重复块。
    """
    seen_keys: Set[str] = set()
    unique_blocks: List[FunctionalBlock] = []

    n = len(code_lines)

    for b in blocks:
        if not b.line_numbers:
            continue

        # 用本 block 的源码片段作为去重 key
        body_lines: List[str] = []
        for ln in sorted(set(b.line_numbers)):
            if 1 <= ln <= n:
                body_lines.append(code_lines[ln - 1].rstrip())

        key = "\n".join(body_lines)

        if key in seen_keys:
            # 完全重复，丢弃
            continue

        seen_keys.add(key)
        unique_blocks.append(b)

    return unique_blocks


# -----------------------
# 高层封装：一键“功能块划分”
# -----------------------

def extract_functional_blocks(
    prog_pdg,
    criteria: List[SlicingCriterion],
    ir2ast_stmt: List,
    code_lines: List[str],
    overlap_threshold: float = 0.75,
    min_lines: int = 12,
    max_lines: int = 150,
    min_lines_stage: int = 8,
) -> List[FunctionalBlock]:
    """
    基于多种切片准则挖掘功能块。

    相比旧版本的主要变化：
      1. 按 criterion.kind 分组后分别调用 cluster_slices，避免不同类型的切片彼此“吃掉”；
      2. 默认 overlap_threshold 调高到 0.75，min_lines 降到 12，偏向生成更多的块；
      3. split_blocks_by_stage 的 stage_var_names 更通用：包含 stage/state/step/phase/mode。

    其它 pipeline（补 IF 结构、按 stage 切分、尺寸归一、去重）保持不变。
    """

    # 0) 基于 ir2ast_stmt 构造 parent_map（只做一次）
    parent_map = build_parent_map_from_ir2ast(ir2ast_stmt)

    # 1) 先对每个准则做一次 PDG 切片
    all_slices: List[Tuple[SlicingCriterion, Set[int]]] = []
    for crit in criteria:
        nodes = compute_slice_nodes(prog_pdg, crit.node_id)
        if not nodes:
            continue
        all_slices.append((crit, nodes))

    if not all_slices:
        return []

    # 1.1 按 kind 分组切片，避免过度混合
    grouped: Dict[str, List[Tuple[SlicingCriterion, Set[int]]]] = defaultdict(list)
    for crit, nodes in all_slices:
        kind = crit.kind or "unknown"
        grouped[kind].append((crit, nodes))

    # 1.2 对每一类 kind 分别调用 cluster_slices，再合并结果
    clusters: List[Dict[str, Any]] = []
    for kind, slices in grouped.items():
        # 你现有的 cluster_slices 接口：cluster_slices(all_slices, overlap_threshold)
        kind_clusters = cluster_slices(slices, overlap_threshold=overlap_threshold)
        # 可以在 cluster 里标记一下 kind，便于 debug
        for c in kind_clusters:
            c.setdefault("kind", kind)
        clusters.extend(kind_clusters)

    # 2) 根据每个 cluster 构建 FunctionalBlock（先不按 stage 切割）
    blocks: List[FunctionalBlock] = []
    for cluster in clusters:
        node_ids: Set[int] = cluster["nodes"]
        crits: List[SlicingCriterion] = cluster["criteria"]

        stmts = nodes_to_sorted_ast_stmts(node_ids, ir2ast_stmt, parent_map)

        # (2.1) 基本行号
        base_lines = stmts_to_line_numbers(stmts, code_lines)

        # (2.2) 补 IF 结构（补 ELSIF/ELSE/END_IF）
        fixed_lines = patch_if_structure(base_lines, code_lines, ensure_end_if=True)

        line_numbers = sorted(fixed_lines)

        block = FunctionalBlock()
        block.criteria = crits
        block.node_ids = set(node_ids)
        block.stmts = stmts
        block.line_numbers = line_numbers
        blocks.append(block)

    # 3) 按 stage/state/step/phase/mode 划分子块
    blocks = split_blocks_by_stage(
        blocks,
        ir2ast_stmt=ir2ast_stmt,
        code_lines=code_lines,
        parent_map=parent_map,
        min_lines=min_lines_stage,
        stage_var_names=("stage", "Stage", "state", "State", "step", "Step", "phase", "Phase", "mode", "Mode"),
    )

    # 4) 尺寸归一：控制块的长度范围在 [min_lines, max_lines]
    blocks = normalize_block_sizes(
        blocks,
        ir2ast_stmt=ir2ast_stmt,
        code_lines=code_lines,
        min_lines=min_lines,
        max_lines=max_lines,
        parent_map=parent_map,
    )

    # 5) 过滤无意义块
    blocks = [b for b in blocks if is_meaningful_block(b, code_lines)]

    # 6) 删除块内空 IF（只改 block.line_numbers，不改 code_lines）
    blocks = remove_empty_ifs_in_blocks(blocks, code_lines)

    # 7) 对完全重复的块按源码去重
    blocks = dedup_blocks_by_code(blocks, code_lines)

    return blocks

