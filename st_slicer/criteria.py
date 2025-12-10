# # st_slicer/criteria.py

# from __future__ import annotations

# from dataclasses import dataclass
# from typing import List, Dict, Set, Optional

# from .slicer import SlicingCriterion, SliceKind
# from .pdg.pdg_builder import ProgramDependenceGraph
# from .dataflow.def_use import DefUseResult
# from .dataflow.var_access import VarAccess
# from .sema.symbols import SymbolTable  # 具体实现由你的工程决定


# @dataclass
# class CriterionConfig:
#     """
#     一些可配置的启发式规则。
#     这里只做名字级别的启发，真正的数据/控制依赖全部走 PDG + DefUse。
#     """
#     error_name_keywords: Optional[List[str]] = None
#     state_name_keywords: Optional[List[str]] = None
#     api_name_prefixes: Optional[List[str]] = None

#     def __post_init__(self):
#         if self.error_name_keywords is None:
#             self.error_name_keywords = ["error", "err", "alarm", "diag", "status"]
#         if self.state_name_keywords is None:
#             self.state_name_keywords = ["stage", "state", "mode", "step", "phase"]
#         if self.api_name_prefixes is None:
#             # 可以按项目再加，比如 ["MC_", "TCP_", "LOG_"]
#             self.api_name_prefixes = ["MC_"]


# # ============ 工具函数 ============

# def _match_any_keyword(name: str, keywords: List[str]) -> bool:
#     s = name.lower()
#     return any(kw.lower() in s for kw in keywords)


# def _collect_node_var_accesses(
#     du: DefUseResult,
#     node_id: int,
# ) -> Dict[str, Set[VarAccess] | Set[str]]:
#     """
#     把某个 PDG 节点（=IR 指令索引）的变量访问信息打包一下，方便后面使用。

#     返回:
#         {
#            "def_vars": set[str],
#            "use_vars": set[str],
#            "def_accesses": set[VarAccess],
#            "use_accesses": set[VarAccess],
#         }
#     """
#     def_vars: Set[str] = set()
#     use_vars: Set[str] = set()
#     def_acc: Set[VarAccess] = set()
#     use_acc: Set[VarAccess] = set()

#     if 0 <= node_id < len(du.def_vars):
#         def_vars = du.def_vars[node_id]
#         use_vars = du.use_vars[node_id]
#     if hasattr(du, "def_accesses") and 0 <= node_id < len(du.def_accesses):
#         def_acc = du.def_accesses[node_id]
#     if hasattr(du, "use_accesses") and 0 <= node_id < len(du.use_accesses):
#         use_acc = du.use_accesses[node_id]

#     return {
#         "def_vars": def_vars,
#         "use_vars": use_vars,
#         "def_accesses": def_acc,
#         "use_accesses": use_acc,
#     }


# # ============ 1. 输出变量相关准则 ============

# def discover_io_output_criteria(
#     symtab: SymbolTable,
#     pdg: ProgramDependenceGraph,
#     du: DefUseResult,
# ) -> List[SlicingCriterion]:
#     """
#     I/O 输出准则：
#       - 先从符号表中找出“输出变量”（输出、InOut、全局输出等）；
#       - 再在 PDG / DefUse 中找所有对这些变量进行定义的指令；
#       - 每个命中的 PDG 节点生成一个 SlicingCriterion。
#     """
#     criteria: List[SlicingCriterion] = []

#     # 1) 找出输出变量名集合；这里不假定具体 SymbolTable 结构，
#     #    只尝试用 get_all_symbols() + sym.role，如果没有，可按自己项目调整。
#     output_vars: Set[str] = set()
#     output_vars = set()

#     for name, var in symtab.vars.items():
#         storage = (var.storage or "").upper()
#         if "OUTPUT" in storage or "IN_OUT" in storage:
#             output_vars.add(name)

#     # # 如果项目暂时没有 role，可以退化为：storage 在 VAR_OUTPUT / VAR_IN_OUT 的变量
#     # if not output_vars and callable(get_all):
#     #     for sym in get_all():
#     #         storage = getattr(sym, "storage", "")
#     #         if storage.upper() in ("VAR_OUTPUT", "VAR_IN_OUT"):
#     #             output_vars.add(sym.name)

#     if not output_vars:
#         return criteria

#     # 2) 遍历 PDG 节点，结合 DefUseResult 找出所有定义输出变量的指令
#     for node_id, node in pdg.nodes.items():
#         info = _collect_node_var_accesses(du, node_id)
#         def_vars = info["def_vars"]
#         def_acc = info["def_accesses"]

#         # 2.1 先看结构化 VarAccess（数组/结构字段）
#         for va in def_acc:
#             if va.base in output_vars:
#                 crit = SlicingCriterion(
#                     node_id=node_id,
#                     kind="io_output",
#                     variable=va.pretty(),  # 例如 axis[1]、pt.X
#                     extra={
#                         "io_base": va.base,
#                         "var_access": va,
#                     },
#                 )
#                 criteria.append(crit)

#         # 2.2 如果没有结构化信息，就回退到字符串变量名
#         if not def_acc:
#             for v in def_vars:
#                 if v in output_vars:
#                     crit = SlicingCriterion(
#                         node_id=node_id,
#                         kind="io_output",
#                         variable=v,
#                         extra={"io_base": v},
#                     )
#                     criteria.append(crit)

#     return criteria


# # ============ 2. 状态变量 & 状态迁移准则 ============

# def discover_state_variables(
#     symtab: SymbolTable,
#     du: DefUseResult,
#     config: CriterionConfig,
# ) -> Set[str]:
#     """
#     自动挖掘“状态变量”：
#       - 名字启发：变量名里包含 state/mode/step/phase 等关键词；
#       - 使用模式启发：在程序中被多次定义 & 多次使用的变量。
#     """
#     state_candidates: Set[str] = set()

#     for name, var in symtab.vars.items():
#         if name and _match_any_keyword(name, config.state_name_keywords):
#             state_candidates.add(name)

#     # 使用模式：多次定义、多次使用
#     var2defs: Dict[str, Set[int]] = getattr(du, "var2defs", {}) or {}
#     use_counter: Dict[str, int] = {}

#     for uses in du.use_vars:
#         for v in uses:
#             use_counter[v] = use_counter.get(v, 0) + 1

#     for v, def_indices in var2defs.items():
#         def_cnt = len(def_indices)
#         use_cnt = use_counter.get(v, 0)
#         if def_cnt >= 2 and use_cnt >= 2:
#             state_candidates.add(v)

#     return state_candidates


# def discover_state_transition_criteria(
#     state_vars: Set[str],
#     pdg: ProgramDependenceGraph,
#     du: DefUseResult,
# ) -> List[SlicingCriterion]:
#     """
#     对每个状态变量的“赋值点”生成切片准则，表示一次状态迁移。
#     利用 DefUseResult 的 def_vars / def_accesses 来定位定义。
#     """
#     criteria: List[SlicingCriterion] = []
#     if not state_vars:
#         return criteria

#     for node_id, node in pdg.nodes.items():
#         info = _collect_node_var_accesses(du, node_id)
#         def_vars = info["def_vars"]
#         def_acc = info["def_accesses"]

#         # 结构化：state_array[1] 这种
#         for va in def_acc:
#             if va.base in state_vars:
#                 criteria.append(
#                     SlicingCriterion(
#                         node_id=node_id,
#                         kind="state_transition",
#                         variable=va.pretty(),
#                         extra={
#                             "state_var": va.base,
#                             "var_access": va,
#                         },
#                     )
#                 )

#         # 非结构化：普通标量 state
#         if not def_acc:
#             for v in def_vars:
#                 if v in state_vars:
#                     criteria.append(
#                         SlicingCriterion(
#                             node_id=node_id,
#                             kind="state_transition",
#                             variable=v,
#                             extra={"state_var": v},
#                         )
#                     )

#     return criteria


# # ============ 3. 错误 / 报警相关准则 ============

# def discover_error_criteria(
#     symtab: SymbolTable,
#     pdg: ProgramDependenceGraph,
#     du: DefUseResult,
#     config: CriterionConfig,
# ) -> List[SlicingCriterion]:
#     """
#     错误/报警准则：
#       - 先基于名字挖掘“错误/报警类变量”集合；
#       - 再在 PDG / DefUse 中找所有定义/使用这些变量的指令。
#     """
#     criteria: List[SlicingCriterion] = []

#     # 1) 名字启发：先找出一批 error-like 变量名
#     error_vars: Set[str] = set()
#     # symtab = POUSymbolTable
#     output_vars = set()

#     for name, var in symtab.vars.items():
#         storage = (var.storage or "").upper()
#         if "OUTPUT" in storage or "IN_OUT" in storage:
#             output_vars.add(name)

#     # 补充：在 DefUse 中出现过的变量里也找一遍
#     for v in getattr(du, "var2defs", {}) or {}:
#         if _match_any_keyword(v, config.error_name_keywords):
#             error_vars.add(v)

#     if not error_vars:
#         return criteria

#     # 2) 遍历 PDG 节点，凡是定义/使用 error_vars 的，都认为是错误/报警逻辑的一部分
#     for node_id, node in pdg.nodes.items():
#         info = _collect_node_var_accesses(du, node_id)
#         def_vars = info["def_vars"]
#         use_vars = info["use_vars"]
#         def_acc = info["def_accesses"]
#         use_acc = info["use_accesses"]

#         matched_var_names: Set[str] = set()
#         matched_accesses: Set[VarAccess] = set()

#         # 2.1 结构化匹配： base 在 error_vars
#         for va in def_acc | use_acc:
#             if va.base in error_vars:
#                 matched_var_names.add(va.base)
#                 matched_accesses.add(va)

#         # 2.2 字符串匹配
#         for v in def_vars | use_vars:
#             if v in error_vars:
#                 matched_var_names.add(v)

#         if not matched_var_names and not matched_accesses:
#             continue

#         # 生成准则（简单起见：每个 node_id 只生成一个 criterion，里面放所有相关变量信息）
#         criteria.append(
#             SlicingCriterion(
#                 node_id=node_id,
#                 kind="error_logic",
#                 variable=", ".join(sorted(matched_var_names)) if matched_var_names else None,
#                 extra={
#                     "error_vars": sorted(matched_var_names),
#                     "error_accesses": [va.pretty() for va in matched_accesses],
#                 },
#             )
#         )

#     return criteria


# # ============ 4. 关键 API 调用准则 ============

# def discover_api_call_criteria(
#     pdg: ProgramDependenceGraph,
#     du: DefUseResult,
#     config: CriterionConfig,
# ) -> List[SlicingCriterion]:
#     """
#     关键 API 调用准则：
#       - 不再依赖 AST，而是直接在 PDG 节点绑定的 IR 指令里找 IRCall；
#       - 根据 callee 名字和 config.api_name_prefixes 判断是否是“关键 API”。
#     """
#     from .ir.ir_nodes import IRCall  # 局部导入避免循环

#     criteria: List[SlicingCriterion] = []

#     for node_id, node in pdg.nodes.items():
#         instr = node.ast_node  # build_program_dependence_graph 里绑定的是 IR 指令
#         if not isinstance(instr, IRCall):
#             continue

#         callee = getattr(instr, "callee", "")
#         if not callee:
#             continue

#         if not any(callee.startswith(prefix) for prefix in config.api_name_prefixes):
#             continue

#         # 关键 API 调用：可以把参数使用的变量也记录下来
#         info = _collect_node_var_accesses(du, node_id)
#         arg_accesses = info["use_accesses"]

#         criteria.append(
#             SlicingCriterion(
#                 node_id=node_id,
#                 kind="api_call",
#                 variable=callee,
#                 extra={
#                     "callee": callee,
#                     "arg_accesses": [va.pretty() for va in arg_accesses],
#                 },
#             )
#         )

#     return criteria


# # ============ 5. 总入口：挖掘所有准则 ============

# def mine_slicing_criteria(
#     pdg: ProgramDependenceGraph,
#     symtab: SymbolTable,
#     du_result: DefUseResult,
#     config: Optional[CriterionConfig] = None,
# ) -> List[SlicingCriterion]:
#     """
#     整体准则挖掘流程：
#       - I/O 输出
#       - 状态机变量的状态切换
#       - 错误/报警逻辑
#       - 关键库函数 / API 调用

#     所有准则都基于：
#       - PDG（数据/控制依赖结构）
#       - DefUseResult（def_vars/use_vars + VarAccess）
#     尽量避免重复在 AST 上做遍历。
#     """
#     if config is None:
#         config = CriterionConfig()

#     criteria: List[SlicingCriterion] = []

#     # 1. I/O 输出
#     criteria.extend(discover_io_output_criteria(symtab, pdg, du_result))

#     # 2. 状态机
#     state_vars = discover_state_variables(symtab, du_result, config)
#     criteria.extend(discover_state_transition_criteria(state_vars, pdg, du_result))

#     # 3. 错误/报警
#     criteria.extend(discover_error_criteria(symtab, pdg, du_result, config))

#     # 4. 关键 API 调用
#     criteria.extend(discover_api_call_criteria(pdg, du_result, config))

#     # TODO：可选去重（同一 node_id 可能被多种规则命中）
#     return criteria

# st_slicer/criteria.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Iterable

from .slicer import SlicingCriterion
from .pdg.pdg_builder import ProgramDependenceGraph
from .dataflow.def_use import DefUseResult
from .dataflow.var_access import VarAccess
from .sema.symbols import POUSymbolTable


# ============ 配置 ============

@dataclass
class CriterionConfig:
    """
    一些可配置的启发式规则
    """
    error_name_keywords: Optional[List[str]] = None
    state_name_keywords: Optional[List[str]] = None
    api_name_prefixes: Optional[List[str]] = None

    def __post_init__(self):
        if self.error_name_keywords is None:
            self.error_name_keywords = ["error", "err", "alarm", "diag", "status", "fault"]
        if self.state_name_keywords is None:
            self.state_name_keywords = ["stage", "state", "mode", "step", "phase"]
        if self.api_name_prefixes is None:
            # 后续可以根据项目再加，比如 ["MC_", "TCP_", "LOG_"]
            self.api_name_prefixes = ["MC_"]


# ============ 一些小工具 ============

def _match_any_keyword(name: str, keywords: Iterable[str]) -> bool:
    lower = name.lower()
    return any(kw in lower for kw in keywords)


def _pretty_access_or_var(
    va: Optional[VarAccess],
    vname: Optional[str],
) -> str:
    """
    把 VarAccess 转成字符串，若没有结构化访问就退回到变量名。
    """
    if va is not None:
        return va.pretty()
    if vname is not None:
        return vname
    return "<unknown>"


def _collect_node_var_accesses(
    node_id: int,
    du: DefUseResult,
) -> Dict[str, Set[VarAccess]]:
    """
    方便使用的一个包装：
      返回该节点上结构化 DEF/USE 访问集合，按 base 聚合。

    返回:
      {
        "def": {VarAccess, ...},
        "use": {VarAccess, ...}
      }
    """
    def_acc = du.def_accesses[node_id] if node_id < len(du.def_accesses) else set()
    use_acc = du.use_accesses[node_id] if node_id < len(du.use_accesses) else set()
    return {"def": set(def_acc), "use": set(use_acc)}


# ============ 1. I/O 输出准则 ============

def discover_io_output_criteria(
    symtab: POUSymbolTable,
    pdg: ProgramDependenceGraph,
    du: DefUseResult,
) -> List[SlicingCriterion]:
    """
    利用符号表 + DefUse：
      - 找出所有“输出角色”的变量（基于 storage/role）
      - 对 PDG 中「定义」这些变量的每个节点生成一个切片准则

    variable 字段：
      - 对普通标量变量：就是变量名本身
      - 对数组/结构成员：使用 VarAccess.pretty()，例如 "axis[1]"、"pt.X"
    """
    criteria: List[SlicingCriterion] = []

    # 1) 从 POUSymbolTable 中找输出 / IN_OUT / 全局输出变量
    output_bases: Set[str] = set()
    storage_map: Dict[str, str] = {}

    for name, sym in symtab.vars.items():
        storage = getattr(sym, "storage", "") or ""
        storage_map[name] = storage
        storage_upper = storage.upper()
        # 你后续可以按项目再精细区分
        if "OUTPUT" in storage_upper or "IN_OUT" in storage_upper:
            output_bases.add(name)

    if not output_bases:
        return criteria

    # 2) 遍历 PDG 节点 (= 指令 index)，看哪些节点定义了这些输出变量
    for node_id in pdg.nodes.keys():
        # a) 先看字符串级 def_vars
        for v in du.def_vars[node_id]:
            if v in output_bases:
                criteria.append(
                    SlicingCriterion(
                        node_id=node_id,
                        kind="io_output",
                        variable=v,
                        extra={
                            "storage": storage_map.get(v, ""),
                            "base": v,
                            "access": None,
                        },
                    )
                )

        # b) 再看结构化访问 def_accesses
        acc = _collect_node_var_accesses(node_id, du)
        for va in acc["def"]:
            if va.base in output_bases:
                criteria.append(
                    SlicingCriterion(
                        node_id=node_id,
                        kind="io_output",
                        variable=va.pretty(),  # 细粒度到 axis[1] / pt.X
                        extra={
                            "storage": storage_map.get(va.base, ""),
                            "base": va.base,
                            "access": va,
                        },
                    )
                )

    return criteria


# ============ 2. 状态变量 & 状态切换准则 ============

def discover_state_variables(
    symtab: POUSymbolTable,
    du: DefUseResult,
    config: CriterionConfig,
) -> Set[str]:
    """
    识别“像状态变量”的候选。

    这里做一个简单的版本（不再遍历 AST，直接用名字 + DefUse 信息）：
      - 名字中包含 state_name_keywords 之一
      - 并且在 DefUse 中有多次定义和使用
    """
    candidates: Set[str] = set()

    # 1) 名字启发式
    for name in symtab.vars.keys():
        if _match_any_keyword(name, config.state_name_keywords):
            candidates.add(name)

    if not candidates:
        return candidates

    # 2) 使用模式启发式：统计 def 次数 / use 次数
    def_count: Dict[str, int] = {}
    use_count: Dict[str, int] = {}

    for i in range(len(du.def_vars)):
        for v in du.def_vars[i]:
            def_count[v] = def_count.get(v, 0) + 1
        for v in du.use_vars[i]:
            use_count[v] = use_count.get(v, 0) + 1

    result: Set[str] = set()
    for v in candidates:
        if def_count.get(v, 0) >= 2 and use_count.get(v, 0) >= 2:
            result.add(v)

    # 如果过滤太狠，至少保留名字启发式的集合
    return result or candidates


def discover_state_transition_criteria(
    state_vars: Set[str],
    pdg: ProgramDependenceGraph,
    du: DefUseResult,
) -> List[SlicingCriterion]:
    """
    对每个状态变量 v 的每个“赋值/定义语句节点”生成准则。

    同样用 DefUse：如果该节点的 def_vars 或 def_accesses 里出现了某状态变量，
    就认为这里是一次潜在的“状态切换”。
    """
    criteria: List[SlicingCriterion] = []

    if not state_vars:
        return criteria

    for node_id in pdg.nodes.keys():
        # 标量层面
        for v in du.def_vars[node_id]:
            if v in state_vars:
                criteria.append(
                    SlicingCriterion(
                        node_id=node_id,
                        kind="state_transition",
                        variable=v,
                        extra={
                            "base": v,
                            "access": None,
                        },
                    )
                )

        # 结构化层面
        acc = _collect_node_var_accesses(node_id, du)
        for va in acc["def"]:
            if va.base in state_vars:
                criteria.append(
                    SlicingCriterion(
                        node_id=node_id,
                        kind="state_transition",
                        variable=va.pretty(),
                        extra={
                            "base": va.base,
                            "access": va,
                        },
                    )
                )

    return criteria


# ============ 3. 错误 / 报警准则 ============

def discover_error_criteria(
    symtab: POUSymbolTable,
    pdg: ProgramDependenceGraph,
    du: DefUseResult,
    config: CriterionConfig,
) -> List[SlicingCriterion]:
    """
    利用符号名 + DefUse 找“错误/报警相关变量”的定义点。
    """
    criteria: List[SlicingCriterion] = []

    # 1) 根据变量名识别 error / alarm 类变量
    error_bases: Set[str] = set()
    for name in symtab.vars.keys():
        if _match_any_keyword(name, config.error_name_keywords):
            error_bases.add(name)

    if not error_bases:
        return criteria

    # 2) 遍历 PDG 节点，找这些变量的定义点
    for node_id in pdg.nodes.keys():
        # 标量
        for v in du.def_vars[node_id]:
            if v in error_bases:
                criteria.append(
                    SlicingCriterion(
                        node_id=node_id,
                        kind="error_logic",
                        variable=v,
                        extra={
                            "base": v,
                            "access": None,
                        },
                    )
                )

        # 结构化访问
        acc = _collect_node_var_accesses(node_id, du)
        for va in acc["def"]:
            if va.base in error_bases:
                criteria.append(
                    SlicingCriterion(
                        node_id=node_id,
                        kind="error_logic",
                        variable=va.pretty(),
                        extra={
                            "base": va.base,
                            "access": va,
                        },
                    )
                )

    return criteria


# ============ 4. 关键 API 调用准则（占位，后面你可以按 AST / IRCall 再细化） ============

def discover_api_call_criteria(
    pdg: ProgramDependenceGraph,
    config: CriterionConfig,
) -> List[SlicingCriterion]:
    """
    目前先留空实现（返回 []），因为你的重点在变量/VarAccess 上。
    后面如果想针对 MC_* 之类的 FB/函数调用做切片，
    可以在 PDG 节点里挂 IRCall / CallStmt，再在这里识别。
    """
    return []


# ============ 5. 总入口：挖掘所有准则 ============

def mine_slicing_criteria(
    pdg: ProgramDependenceGraph,
    du: DefUseResult,
    symtab: POUSymbolTable,
    config: Optional[CriterionConfig] = None,
) -> List[SlicingCriterion]:
    """
    整体准则挖掘流程（已经切换到基于 DefUse + VarAccess 的变量敏感版）：

      - I/O 输出：使用输出变量的定义点（支持数组/结构成员）
      - 状态变量：根据名字 + DefUse 粗识别，再基于定义点生成状态切换准则
      - 错误/报警：名字匹配 + 定义点
      - 关键 API 调用：目前留空（可在后续补充）
    """
    if config is None:
        config = CriterionConfig()

    criteria: List[SlicingCriterion] = []

    # 1. I/O 输出
    criteria.extend(discover_io_output_criteria(symtab, pdg, du))

    # 2. 状态机
    state_vars = discover_state_variables(symtab, du, config)
    criteria.extend(discover_state_transition_criteria(state_vars, pdg, du))

    # 3. 错误/报警
    criteria.extend(discover_error_criteria(symtab, pdg, du, config))

    # 4. 关键 API 调用（目前为空实现）
    criteria.extend(discover_api_call_criteria(pdg, config))

    # TODO：可以按 node_id + variable 去重
    return criteria
