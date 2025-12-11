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

def classify_variable(symbol):
    """
    根据 symbol 判断变量应该放在什么声明区。
    返回:
        ("fb_instance", type_name)   → 需要放 VAR 中的 FB 实例
        ("normal_var", type_name)    → 普通变量
        ("ignore", None)             → function，不需要声明
    """
    role = getattr(symbol, "role", None)
    typ  = getattr(symbol, "type", None)

    # FB 实例
    if role in ("FB", "FUNCTION_BLOCK"):
        return ("fb_instance", typ)

    # function 调用：不出现在变量区
    if role in ("FUNCTION",):
        return ("ignore", None)
        
    return ("normal_var", typ)


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

    # 1) 收集块中使用到的变量名（只来自 AST / 语句）
    vars_used: Set[str] = collect_vars_in_block(block.stmts)

    # 2) 构建 name -> symbol 的映射
    sym_by_name: Dict[str, object] = {
        sym.name: sym
        for sym in pou_symtab.get_all_symbols()
    }

    # 各类声明收集
    fb_instance_decls: List[str] = []   # 功能块实例，例如 FB_S_Type_...
    var_input_decls: List[str] = []     # VAR_INPUT
    var_output_decls: List[str] = []    # VAR_OUTPUT
    var_local_decls: List[str] = []     # 普通 VAR
    # ★ 不再维护 stub_decls

    # 一些“明显是函数/系统功能，而非变量或 FB 实例”的名字，可以直接忽略
    known_func_like_prefixes = ("MC_", "F_", "REAL_TO_", "UDINT_TO_", "DINT_TO_")
    known_func_like_names = {
        "ESQR",
        "RealAbs",
        "UDINT_TO_REAL",
        "DINT_TO_REAL",
    }

    # 3) 逐个变量分类
    for v in sorted(vars_used):
        sym = sym_by_name.get(v)

        # 3.1 在符号表中找不到
        if sym is None:
            # 像 MC_RdAxisPar_FL、系统函数一类的：直接忽略，不声明
            if v.upper().startswith(known_func_like_prefixes) or v in known_func_like_names:
                continue

            # ★ 关键修改：其它未知名字也不再兜底生成 stub 变量
            # 认为它可能是全局变量 / 外部库变量 / 常量，由工程环境解决
            continue

        # 3.2 在符号表中找得到，判断它的 “角色” 和 “存储类别”
        storage = (getattr(sym, "storage", "") or "").upper()
        v_type = getattr(sym, "type", "REAL")

        # role/kind 可能在不同实现里字段名不一样，这里都尝试一下
        role = (
            (getattr(sym, "role", "") or "")
            or (getattr(sym, "kind", "") or "")
        ).upper()

        # 3.2.1 功能块实例（FB instance），需要放在 VAR 区里： name : FBType;
        if role in ("FB", "FUNCTION_BLOCK", "FB_INSTANCE"):
            fb_instance_decls.append(f"    {sym.name} : {v_type};")
            continue

        # 3.2.2 函数 / 程序 / 方法之类：不需要在本块里声明
        if role in ("FUNCTION", "FUNC", "METHOD", "ACTION", "PROGRAM"):
            continue

        # 3.2.3 普通变量，根据 storage 放到 VAR_INPUT / VAR_OUTPUT / VAR
        decl = f"    {sym.name} : {v_type};"

        if storage == "VAR_INPUT":
            var_input_decls.append(decl)
        elif storage == "VAR_OUTPUT":
            var_output_decls.append(decl)
        else:
            var_local_decls.append(decl)

    # 4) 组装 PROGRAM 框架
    prog_name = f"{pou_name}_BLOCK_{block_index}"
    out_lines: List[str] = []

    out_lines.append(f"PROGRAM {prog_name}")

    # 4.1 功能块实例（FB 实例）放在一个 VAR 区（如果你想单独分区也可以调整）
    if fb_instance_decls:
        out_lines.append("VAR")
        out_lines.extend(fb_instance_decls)
        out_lines.append("END_VAR")

    # 4.2 VAR_INPUT 区
    if var_input_decls:
        out_lines.append("VAR_INPUT")
        out_lines.extend(var_input_decls)
        out_lines.append("END_VAR")

    # 4.3 VAR_OUTPUT 区
    if var_output_decls:
        out_lines.append("VAR_OUTPUT")
        out_lines.extend(var_output_decls)
        out_lines.append("END_VAR")

    # 4.4 普通 VAR（不再拼接 stub_decls）
    if var_local_decls:
        out_lines.append("VAR")
        out_lines.extend(var_local_decls)
        out_lines.append("END_VAR")

    out_lines.append("")

    # 5) 功能块主体代码：从原源码按行号直接拷贝
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
