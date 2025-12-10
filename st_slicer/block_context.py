# st_slicer/block_context.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Dict

from .functional_blocks import FunctionalBlock, collect_vars_in_block


@dataclass
class CompletedBlock:
    """
    表示一个已经“结构补全”的功能块：
      - code: 完整的 ST 程序文本（含 PROGRAM/VAR/END_PROGRAM）
      - line_numbers: 源程序中涉及到的行号
      - vars_used: 该块中用到的变量名集合
    """
    block_index: int
    code: str
    line_numbers: List[int]
    vars_used: Set[str]


def _classify_var_storage(sym) -> str:
    """
    根据符号表中的 symbol 判断变量的存储类别：
      - VAR_INPUT / VAR_OUTPUT / VAR（或其他）
    你目前的 symbol 好像有 type / role，可以兼容 storage/role 两种字段。
    """
    storage = getattr(sym, "storage", None)
    if storage is None:
        storage = getattr(sym, "role", None)

    if storage is None:
        # 默认按普通 VAR 处理
        return "VAR"

    storage_upper = str(storage).upper()
    if "INPUT" in storage_upper:
        return "VAR_INPUT"
    if "OUTPUT" in storage_upper:
        return "VAR_OUTPUT"
    return "VAR"


def build_completed_block(
    block: FunctionalBlock,
    pou_name: str,
    pou_symtab,
    code_lines: List[str],
    block_index: int,
) -> CompletedBlock:
    """
    根据一个 FunctionalBlock + 原 POU 的符号表，生成一个完整可编译的 ST 程序代码串。

    参数:
      - block: 功能块（包含 node_ids / stmts / line_numbers）
      - pou_name: 原 POU 名字，用来生成 PROGRAM 名
      - pou_symtab: 符号表，对应 build_symbol_table(...).get_pou(pou_name)
      - code_lines: 原 ST 源码的按行列表
      - block_index: 当前块索引，用于 PROGRAM 名后缀
    """

    # 1) 收集块中使用到的变量名
    vars_used: Set[str] = collect_vars_in_block(block.stmts)

    # 2) 建立 name -> symbol 的映射，方便查找
    sym_by_name: Dict[str, object] = {
        sym.name: sym
        for sym in pou_symtab.get_all_symbols()
    }

    var_input: List[str] = []
    var_output: List[str] = []
    var_local: List[str] = []
    var_missing: List[str] = []  # 在符号表中找不到的变量，生成 stub

    for v in sorted(vars_used):
        sym = sym_by_name.get(v)
        if sym is None:
            # 在原 POU 中没找到，后面生成 stub
            var_missing.append(v)
            continue

        storage_class = _classify_var_storage(sym)
        v_type = getattr(sym, "type", "REAL")  # 没有类型就兜底 REAL
        decl = f"    {sym.name} : {v_type};"

        if storage_class == "VAR_INPUT":
            var_input.append(decl)
        elif storage_class == "VAR_OUTPUT":
            var_output.append(decl)
        else:
            var_local.append(decl)

    # 3) 组装 PROGRAM 框架
    prog_name = f"{pou_name}_BLOCK_{block_index}"
    out_lines: List[str] = []

    out_lines.append(f"PROGRAM {prog_name}")

    # 3.1 VAR_INPUT 区
    if var_input:
        out_lines.append("VAR_INPUT")
        out_lines.extend(var_input)
        out_lines.append("END_VAR")

    # 3.2 VAR_OUTPUT 区
    if var_output:
        out_lines.append("VAR_OUTPUT")
        out_lines.extend(var_output)
        out_lines.append("END_VAR")

    # 3.3 VAR 区：普通局部变量 + stub 变量
    if var_local or var_missing:
        out_lines.append("VAR")
        out_lines.extend(var_local)

        # stub 变量：在原符号表里没找到类型的变量
        for v in var_missing:
            out_lines.append(
                f"    {v} : REAL; // TODO: stub, type unknown in original POU"
            )
        out_lines.append("END_VAR")

    out_lines.append("")

    # 4) 功能块主体代码：从原源码按行号拷贝
    out_lines.append("(* ===== Functional body from original code ===== *)")
    for ln in sorted(block.line_numbers):
        if 1 <= ln <= len(code_lines):
            out_lines.append(code_lines[ln - 1].rstrip())

    out_lines.append("END_PROGRAM")

    code = "\n".join(out_lines)

    return CompletedBlock(
        block_index=block_index,
        code=code,
        line_numbers=list(block.line_numbers),
        vars_used=vars_used,
    )
