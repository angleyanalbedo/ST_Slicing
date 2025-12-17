"""
criteria.py

通用的 PLC / 运动控制 / 状态机 ST 程序切片准则挖掘模块。

核心设计：
  - 通用层：
      * I/O 输出准则 (io_output)
      * 状态变量准则 (state_transition)
      * 错误逻辑准则 (error_logic)
  - 状态机 / 控制逻辑层：
      * 阶段边界 (stage_boundary)：同时写 state + 输出
      * 运行事件 (runtime_done / runtime_error / runtime_interrupt)
  - 运动 / 物理量层（不限于 MC_*，适用于所有有“位置/速度/时间/限值”等量的程序）：
      * 运动物理量 (motion_quantity)：pos/vel/acc/time/dist/seg 等变量的定义点
      * 特殊场景 (motion_special_case)：运动物理量 + 限值变量 (min/max/limit/tol/threshold) 同时出现的节点
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

#from .slicer import SlicingCriterion
from .pdg.pdg_builder import ProgramDependenceGraph
from .dataflow.def_use import DefUseResult
from .dataflow.var_access import VarAccess
from .sema.symbols import POUSymbolTable
from st_slicer.blocks.types import SlicingCriterion

# ======== 配置：控制 / 运动领域的通用启发式 =========

@dataclass
class CriterionConfig:
    """
    控制 / 运动领域的通用启发式配置。
    所有字段都只是“名字匹配规则”，不依赖具体文件或具体 POU。
    """

    # 错误类变量
    error_name_keywords: Optional[List[str]] = None
    # 状态类变量（状态机 / 模式）
    state_name_keywords: Optional[List[str]] = None
    # API/功能块调用前缀
    api_name_prefixes: Optional[List[str]] = None

    # 轴 / 通道 / 关节等
    axis_name_keywords: Optional[List[str]] = None
    # 运动轮廓/物理量变量
    motion_quantity_keywords: Optional[List[str]] = None
    # 阈值 / 限值变量
    limit_keywords: Optional[List[str]] = None
    # 中断 / 停止 / 退出变量
    interrupt_keywords: Optional[List[str]] = None
    # 完成 / 结束 / 允许变量
    done_keywords: Optional[List[str]] = None

    def __post_init__(self):
        if self.error_name_keywords is None:
            self.error_name_keywords = [
                "error", "err", "alarm", "fault", "diag", "status",
            ]

        if self.state_name_keywords is None:
            self.state_name_keywords = [
                "stage", "state", "step", "phase", "mode",
            ]

        if self.api_name_prefixes is None:
            # 不写死 MC_，但把它作为默认之一
            self.api_name_prefixes = ["MC_", "FB_", "AXIS_", "DRV_"]

        # 轴 / 通道
        if self.axis_name_keywords is None:
            self.axis_name_keywords = [
                "axis", "axes", "joint", "servo", "motor", "channel",
            ]

        # 运动物理量：位置、速度、加速度、时间、距离等
        if self.motion_quantity_keywords is None:
            self.motion_quantity_keywords = [
                "pos", "position", "angle", "dist", "distance",
                "vel", "velocity", "speed",
                "acc", "accel", "decel", "dec", "jerk",
                "time", "t_", "dt",
                "len", "length",
                "count", "cnt",
                "segment", "seg", "section",
            ]

        # 阈值/限值
        if self.limit_keywords is None:
            self.limit_keywords = [
                "min", "max", "limit", "lim", "tol", "threshold",
            ]

        # 中断/停止/取消类
        if self.interrupt_keywords is None:
            self.interrupt_keywords = [
                "interrupt", "abort", "stop", "emergency", "hold",
            ]

        # 完成/结束/允许类
        if self.done_keywords is None:
            self.done_keywords = [
                "done", "complete", "finished", "ready", "enable", "allow",
            ]


# ======== 一些小工具函数 =========

def _match_any_keyword(name: str, keywords: List[str]) -> bool:
    """
    简单的名字匹配：忽略大小写，子串匹配。
    例如 name="Axis_state" 能匹配关键词 "axis" 或 "state"。
    """
    if not name:
        return False
    low = name.lower()
    for kw in keywords:
        if kw and kw.lower() in low:
            return True
    return False


def _collect_node_var_accesses(node_id: int, du: DefUseResult) -> Dict[str, List[VarAccess]]:
    """
    对某个节点，收集结构化变量访问（VarAccess），返回：
      {
        "def": [VarAccess, ...],
        "use": [VarAccess, ...],
      }
    具体属性名 (def_accesses/use_accesses) 需与你自己的 DefUseResult 对齐。
    """
    def_acc = getattr(du, "def_accesses", None)
    use_acc = getattr(du, "use_accesses", None)

    def_list: List[VarAccess] = []
    use_list: List[VarAccess] = []

    if def_acc is not None and 0 <= node_id < len(def_acc):
        def_list = def_acc[node_id] or []
    if use_acc is not None and 0 <= node_id < len(use_acc):
        use_list = use_acc[node_id] or []

    return {"def": def_list, "use": use_list}

def _get_ast_node_for_pdg_node(pdg: ProgramDependenceGraph, node_id: int) -> Any:
    """
    从 PDG 中取出与 node_id 对应的 AST 节点。
    要求 ProgramDependenceGraph 在构建时提供 node_to_ast 映射：
        pdg.node_to_ast: Dict[int, AstNode]

    如果没有这个属性，则返回 None，不影响其它准则。
    """
    mapping = getattr(pdg, "node_to_ast", None)
    if mapping is None:
        return None
    return mapping.get(node_id)


# ======== 通用：I/O 输出准则 =========

def discover_io_output_criteria(
    symtab: POUSymbolTable,
    pdg: ProgramDependenceGraph,
    du: DefUseResult,
) -> List[SlicingCriterion]:
    """
    凡是给 VAR_OUTPUT / VAR_IN_OUT 变量赋值的节点，都视为 io_output 准则。
    """
    criteria: List[SlicingCriterion] = []

    output_bases: Set[str] = set()
    for name, sym in getattr(symtab, "vars", {}).items():
        storage = (getattr(sym, "storage", "") or "").upper()
        if "OUTPUT" in storage or "IN_OUT" in storage:
            output_bases.add(name)

    if not output_bases:
        return criteria

    for node_id in pdg.nodes.keys():
        if node_id < 0 or node_id >= len(du.def_vars):
            continue

        # 标量
        for v in du.def_vars[node_id]:
            if v in output_bases:
                criteria.append(
                    SlicingCriterion(
                        node_id=node_id,
                        kind="io_output",
                        variable=v,
                        extra={"base": v, "access": None},
                    )
                )

        # 结构化
        acc = _collect_node_var_accesses(node_id, du)
        for va in acc["def"]:
            if va.base in output_bases:
                criteria.append(
                    SlicingCriterion(
                        node_id=node_id,
                        kind="io_output",
                        variable=getattr(va, "pretty", lambda: va.base)(),
                        extra={"base": va.base, "access": va},
                    )
                )

    return criteria


# ======== 通用：状态变量及其切换准则 =========

def discover_state_variables(
    symtab: POUSymbolTable,
    du: DefUseResult,
    config: CriterionConfig,
) -> Set[str]:
    """
    候选状态变量 = 名字里包含 stage/state/step/phase/mode 等的变量，
    再用 def/use 次数做一次过滤。
    """
    candidates: Set[str] = set()
    for name in getattr(symtab, "vars", {}).keys():
        if _match_any_keyword(name, config.state_name_keywords or []):
            candidates.add(name)

    if not candidates:
        return candidates

    def_counts: Dict[str, int] = {v: 0 for v in candidates}
    use_counts: Dict[str, int] = {v: 0 for v in candidates}

    for node_id in range(len(du.def_vars)):
        for v in du.def_vars[node_id]:
            if v in def_counts:
                def_counts[v] += 1
        for v in du.use_vars[node_id]:
            if v in use_counts:
                use_counts[v] += 1

    result: Set[str] = set()
    for v in candidates:
        if def_counts[v] >= 2 and use_counts[v] >= 2:
            result.add(v)

    return result or candidates


def discover_state_transition_criteria(
    state_vars: Set[str],
    pdg: ProgramDependenceGraph,
    du: DefUseResult,
) -> List[SlicingCriterion]:
    """
    状态变量的定义点均视为 state_transition 准则。
    """
    criteria: List[SlicingCriterion] = []
    if not state_vars:
        return criteria

    for node_id in pdg.nodes.keys():
        if node_id < 0 or node_id >= len(du.def_vars):
            continue

        has = False
        var_name = None

        for v in du.def_vars[node_id]:
            if v in state_vars:
                has = True
                var_name = v
                break

        acc = _collect_node_var_accesses(node_id, du)
        if not has:
            for va in acc["def"]:
                if va.base in state_vars:
                    has = True
                    var_name = va.base
                    break

        if not has:
            continue

        criteria.append(
            SlicingCriterion(
                node_id=node_id,
                kind="state_transition",
                variable=var_name or "<state>",
                extra={"state_vars": sorted(state_vars)},
            )
        )

    return criteria


# ======== 通用：错误逻辑准则 =========

def discover_error_criteria(
    symtab: POUSymbolTable,
    pdg: ProgramDependenceGraph,
    du: DefUseResult,
    config: CriterionConfig,
) -> List[SlicingCriterion]:
    """
    名字匹配 error/err/alarm/fault/diag/status 的变量的定义点，视为 error_logic。
    """
    criteria: List[SlicingCriterion] = []

    error_bases: Set[str] = set()
    for name in getattr(symtab, "vars", {}).keys():
        if _match_any_keyword(name, config.error_name_keywords or []):
            error_bases.add(name)

    if not error_bases:
        return criteria

    for node_id in pdg.nodes.keys():
        if node_id < 0 or node_id >= len(du.def_vars):
            continue

        def_names = du.def_vars[node_id]
        acc = _collect_node_var_accesses(node_id, du)

        for v in def_names:
            if v in error_bases:
                criteria.append(
                    SlicingCriterion(
                        node_id=node_id,
                        kind="error_logic",
                        variable=v,
                        extra={"base": v, "access": None},
                    )
                )

        for va in acc["def"]:
            if va.base in error_bases:
                criteria.append(
                    SlicingCriterion(
                        node_id=node_id,
                        kind="error_logic",
                        variable=getattr(va, "pretty", lambda: va.base)(),
                        extra={"base": va.base, "access": va},
                    )
                )

    return criteria


# ======== 状态机 / 控制逻辑：阶段边界准则 =========

def discover_stage_boundary_criteria(
    symtab: POUSymbolTable,
    pdg: ProgramDependenceGraph,
    du: DefUseResult,
    config: CriterionConfig,
) -> List[SlicingCriterion]:
    """
    通用“阶段边界”准则：

    节点上同时：
      - 定义状态变量（state/stage/step/mode/phase 等）
      - 定义输出变量（VAR_OUTPUT / VAR_IN_OUT）
    认为这是一个阶段切换边界，kind="stage_boundary"
    """
    criteria: List[SlicingCriterion] = []

    state_vars = discover_state_variables(symtab, du, config)
    if not state_vars:
        return criteria

    output_bases: Set[str] = set()
    for name, sym in getattr(symtab, "vars", {}).items():
        storage = (getattr(sym, "storage", "") or "").upper()
        if "OUTPUT" in storage or "IN_OUT" in storage:
            output_bases.add(name)

    if not output_bases:
        return criteria

    for node_id in pdg.nodes.keys():
        if node_id < 0 or node_id >= len(du.def_vars):
            continue

        def_names = du.def_vars[node_id]
        acc = _collect_node_var_accesses(node_id, du)

        has_state_def = any(v in state_vars for v in def_names) or any(
            va.base in state_vars for va in acc["def"]
        )
        has_output_def = any(v in output_bases for v in def_names) or any(
            va.base in output_bases for va in acc["def"]
        )

        if not (has_state_def and has_output_def):
            continue

        state_name = next((v for v in def_names if v in state_vars), None)
        if state_name is None:
            for va in acc["def"]:
                if va.base in state_vars:
                    state_name = va.base
                    break

        criteria.append(
            SlicingCriterion(
                node_id=node_id,
                kind="stage_boundary",
                variable=state_name or "<state>",
                extra={
                    "state_vars": sorted(state_vars),
                    "output_bases": sorted(output_bases),
                },
            )
        )

    return criteria


# ======== 状态机 / 控制逻辑：运行事件准则 =========

def discover_runtime_event_criteria(
    symtab: POUSymbolTable,
    pdg: ProgramDependenceGraph,
    du: DefUseResult,
    config: CriterionConfig,
) -> List[SlicingCriterion]:
    """
    通用运行阶段事件准则：

    - runtime_done:
        * 节点定义状态变量 AND 定义 "完成/允许" 类输出，如 Done/Ready/Enable
    - runtime_error:
        * 节点定义状态变量 AND 定义 error 类变量
    - runtime_interrupt:
        * 节点定义状态变量 AND use 中包含中断/停止等关键词
    """
    criteria: List[SlicingCriterion] = []

    state_vars = discover_state_variables(symtab, du, config)
    if not state_vars:
        return criteria

    # 输出变量集合
    output_bases: Set[str] = set()
    for name, sym in getattr(symtab, "vars", {}).items():
        storage = (getattr(sym, "storage", "") or "").upper()
        if "OUTPUT" in storage or "IN_OUT" in storage:
            output_bases.add(name)

    # error 变量集合
    error_bases: Set[str] = set()
    for name in getattr(symtab, "vars", {}).keys():
        if _match_any_keyword(name, config.error_name_keywords or []):
            error_bases.add(name)

    # “完成/允许类”输出
    done_like_outputs: Set[str] = set(
        name for name in getattr(symtab, "vars", {}).keys()
        if _match_any_keyword(name, config.done_keywords or [])
    )

    for node_id in pdg.nodes.keys():
        if node_id < 0 or node_id >= len(du.def_vars):
            continue

        def_names = du.def_vars[node_id]
        use_names = du.use_vars[node_id]
        acc = _collect_node_var_accesses(node_id, du)

        has_state_def = any(v in state_vars for v in def_names) or any(
            va.base in state_vars for va in acc["def"]
        )
        if not has_state_def:
            continue

        has_error_def = any(v in error_bases for v in def_names) or any(
            va.base in error_bases for va in acc["def"]
        )
        has_done_def = any(v in done_like_outputs for v in def_names) or any(
            va.base in done_like_outputs for va in acc["def"]
        )

        has_interrupt_use = any(
            _match_any_keyword(v, config.interrupt_keywords or []) for v in use_names
        ) or any(
            _match_any_keyword(va.base, config.interrupt_keywords or []) for va in acc["use"]
        )

        if has_interrupt_use:
            kind = "runtime_interrupt"
        elif has_error_def:
            kind = "runtime_error"
        elif has_done_def:
            kind = "runtime_done"
        else:
            continue

        state_name = next((v for v in def_names if v in state_vars), None)
        if state_name is None:
            for va in acc["def"]:
                if va.base in state_vars:
                    state_name = va.base
                    break

        criteria.append(
            SlicingCriterion(
                node_id=node_id,
                kind=kind,
                variable=state_name or "<state>",
                extra={
                    "state_vars": sorted(state_vars),
                    "error_bases": sorted(error_bases),
                    "done_outputs": sorted(done_like_outputs),
                },
            )
        )

    return criteria

# def discover_control_branch_criteria(
#     pdg: ProgramDependenceGraph,
#     du: DefUseResult,
# ) -> List[SlicingCriterion]:
#     """
#     通用分支准则（control_branch）：

#     - 对每个 IF / ELSIF / CASE 分支头部，对应的 PDG 节点生成一个切片准则。
#     - 仅依赖 AST 结构，不依赖变量名或领域知识。

#     需要：
#       - pdg.node_to_ast: Dict[int, AstNode]
#       - AST 节点类型名中包含:
#             "IfStmt", "ElsifBranch", "CaseStmt", "CaseElement" 等
#       - 如果工程里类型名不同，可把下面的 class_name 判断改成自己的名字。
#     """
#     criteria: List[SlicingCriterion] = []

#     # 没有 AST 映射就直接返回空，避免破坏原有行为
#     if not hasattr(pdg, "node_to_ast"):
#         return criteria

#     for node_id in pdg.nodes.keys():
#         if node_id < 0 or node_id >= len(du.def_vars):
#             continue

#         ast_node = _get_ast_node_for_pdg_node(pdg, node_id)
#         if ast_node is None:
#             continue

#         cls_name = type(ast_node).__name__

#         # IF / ELSIF 分支头
#         if cls_name in ("IfStmt", "ElsifBranch"):
#             criteria.append(
#                 SlicingCriterion(
#                     node_id=node_id,
#                     kind="control_branch",
#                     variable="<if_branch>",
#                     extra={"ast_type": cls_name},
#                 )
#             )
#             continue

#         # CASE / CASE 分支
#         if cls_name in ("CaseStmt", "CaseElement"):
#             criteria.append(
#                 SlicingCriterion(
#                     node_id=node_id,
#                     kind="control_branch",
#                     variable="<case_branch>",
#                     extra={"ast_type": cls_name},
#                 )
#             )
#             continue

#     return criteria

def discover_control_branch_criteria(
    pdg: ProgramDependenceGraph,
    du: DefUseResult,
) -> List[SlicingCriterion]:
    """
    通用分支准则（control_branch）：

    - 对每个 IF / ELSIF / CASE 分支头部，对应的 PDG 节点生成一个切片准则。
    - 仅依赖 AST 结构，不依赖变量名或领域知识。

    需要：
      - pdg.node_to_ast: Dict[int, AstNode]
      - AST 节点类型名中包含:
            "IfStmt", "ElsifBranch", "CaseStmt", "CaseElement" 等
      - 如果工程里类型名不同，可把下面的 class_name 判断改成自己的名字。
    """
    criteria: List[SlicingCriterion] = []

    # 没有 AST 映射就直接返回空，避免破坏原有行为
    if not hasattr(pdg, "node_to_ast"):
        return criteria

    for node_id in pdg.nodes.keys():
        if node_id < 0 or node_id >= len(du.def_vars):
            continue

        ast_node = _get_ast_node_for_pdg_node(pdg, node_id)
        if ast_node is None:
            continue

        cls_name = type(ast_node).__name__

        # IF / ELSIF 分支头
        if cls_name in ("IfStmt", "ElsifBranch"):
            criteria.append(
                SlicingCriterion(
                    node_id=node_id,
                    kind="control_branch",
                    variable="<if_branch>",
                    extra={"ast_type": cls_name},
                )
            )
            continue

        # CASE / CASE 分支
        if cls_name in ("CaseStmt", "CaseElement"):
            criteria.append(
                SlicingCriterion(
                    node_id=node_id,
                    kind="control_branch",
                    variable="<case_branch>",
                    extra={"ast_type": cls_name},
                )
            )
            continue

    return criteria

def discover_loop_region_criteria(
    pdg: ProgramDependenceGraph,
    du: DefUseResult,
) -> List[SlicingCriterion]:
    """
    通用循环准则（loop_region）：

    - 对每个 FOR / WHILE / REPEAT 循环头对应的 PDG 节点生成一个准则。
    - 仅依赖 AST 结构，不依赖领域知识。
    """
    criteria: List[SlicingCriterion] = []

    if not hasattr(pdg, "node_to_ast"):
        return criteria

    for node_id in pdg.nodes.keys():
        if node_id < 0 or node_id >= len(du.def_vars):
            continue

        ast_node = _get_ast_node_for_pdg_node(pdg, node_id)
        if ast_node is None:
            continue

        cls_name = type(ast_node).__name__

        if cls_name in ("ForStmt", "WhileStmt", "RepeatStmt"):
            criteria.append(
                SlicingCriterion(
                    node_id=node_id,
                    kind="loop_region",
                    variable="<loop>",
                    extra={"ast_type": cls_name},
                )
            )

    return criteria

# ======== 运动 / 物理量：运动物理量准则 =========

def discover_motion_quantity_criteria(
    symtab: POUSymbolTable,
    pdg: ProgramDependenceGraph,
    du: DefUseResult,
    config: CriterionConfig,
) -> List[SlicingCriterion]:
    """
    通用“运动物理量”准则：

    - 名字中包含 pos/position/angle/vel/acc/time/dist/seg 等的变量
    - 它们的定义点一律视为 motion_quantity 准则
    """
    criteria: List[SlicingCriterion] = []

    mq_keywords = config.motion_quantity_keywords or []
    if not mq_keywords:
        return criteria

    for node_id in pdg.nodes.keys():
        if node_id < 0 or node_id >= len(du.def_vars):
            continue

        def_names = du.def_vars[node_id]
        acc = _collect_node_var_accesses(node_id, du)

        for v in def_names:
            if _match_any_keyword(v, mq_keywords):
                criteria.append(
                    SlicingCriterion(
                        node_id=node_id,
                        kind="motion_quantity",
                        variable=v,
                        extra={"base": v, "access": None},
                    )
                )

        for va in acc["def"]:
            if _match_any_keyword(va.base, mq_keywords):
                criteria.append(
                    SlicingCriterion(
                        node_id=node_id,
                        kind="motion_quantity",
                        variable=getattr(va, "pretty", lambda: va.base)(),
                        extra={"base": va.base, "access": va},
                    )
                )

    return criteria


# ======== 运动 / 物理量：特殊场景准则 =========

def discover_motion_special_case_criteria(
    symtab: POUSymbolTable,
    pdg: ProgramDependenceGraph,
    du: DefUseResult,
    config: CriterionConfig,
) -> List[SlicingCriterion]:
    """
    通用“特殊场景”准则：

    节点满足：
      - 定义运动物理量（见 motion_quantity_criteria）
      - use 到 limit_keywords 里的变量（min/max/limit/tol/threshold 等）
    视为 motion_special_case 准则
    """
    criteria: List[SlicingCriterion] = []

    mq_keywords = config.motion_quantity_keywords or []
    limit_keywords = config.limit_keywords or []
    if not mq_keywords or not limit_keywords:
        return criteria

    for node_id in pdg.nodes.keys():
        if node_id < 0 or node_id >= len(du.def_vars):
            continue

        def_names = du.def_vars[node_id]
        acc = _collect_node_var_accesses(node_id, du)

        has_mq_def = any(_match_any_keyword(v, mq_keywords) for v in def_names) or any(
            _match_any_keyword(va.base, mq_keywords) for va in acc["def"]
        )
        if not has_mq_def:
            continue

        use_names = du.use_vars[node_id]
        has_limit_use = any(
            _match_any_keyword(v, limit_keywords) for v in use_names
        ) or any(
            _match_any_keyword(va.base, limit_keywords) for va in acc["use"]
        )
        if not has_limit_use:
            continue

        mq_name = next((v for v in def_names if _match_any_keyword(v, mq_keywords)), None)
        if mq_name is None:
            for va in acc["def"]:
                if _match_any_keyword(va.base, mq_keywords):
                    mq_name = va.base
                    break

        criteria.append(
            SlicingCriterion(
                node_id=node_id,
                kind="motion_special_case",
                variable=mq_name or "<motion>",
                extra={"limit_keywords": limit_keywords},
            )
        )

    return criteria


# ======== 关键 API 调用（占位，可按需扩展） =========

def discover_api_call_criteria(
    pdg: ProgramDependenceGraph,
    du: DefUseResult,
    config: CriterionConfig,
) -> List[SlicingCriterion]:
    """
    通用函数/FB 调用准则（api_call）：

    - 对每个调用语句对应的 PDG 节点生成一个准则。
    - 优先使用 config.api_name_prefixes 过滤（MC_/FB_/AXIS_/DRV_），
      如果你想对所有调用都切片，可以去掉前缀过滤。
    - 仅依赖 AST 结构（节点类型名含 Call/FBCall/FuncCall 等，或带 callee/name 属性）。
    """
    criteria: List[SlicingCriterion] = []

    if not hasattr(pdg, "node_to_ast"):
        return criteria

    prefixes = config.api_name_prefixes or []

    for node_id in pdg.nodes.keys():
        if node_id < 0 or node_id >= len(du.def_vars):
            continue

        ast_node = _get_ast_node_for_pdg_node(pdg, node_id)
        if ast_node is None:
            continue

        cls_name = type(ast_node).__name__

        # 用类名粗略判断：CallStmt / CallExpr / FuncCall / FBCall 等
        is_call_like = any(
            key in cls_name
            for key in ("Call", "FBCall", "FuncCall")
        )

        if not is_call_like:
            # 有些项目里，调用节点类名不带 Call，可以再用属性探测
            # 比如有 callee/name 成员的，也可以视作调用节点
            if not (hasattr(ast_node, "callee") or hasattr(ast_node, "name")):
                continue

        # 尝试获取被调用对象名字
        callee_name = None
        if hasattr(ast_node, "name"):
            callee_name = getattr(ast_node, "name", None)
        elif hasattr(ast_node, "callee"):
            callee = getattr(ast_node, "callee", None)
            if hasattr(callee, "name"):
                callee_name = getattr(callee, "name", None)

        if not isinstance(callee_name, str) or not callee_name:
            callee_name = "<call>"

        # 如果你只想对 MC_/FB_/AXIS_/DRV_ 这些调用切片，可以保留前缀过滤；
        # 想对所有调用切片，可以注释掉下面这段 if。
        if prefixes:
            lowered = callee_name.upper()
            if not any(lowered.startswith(p.upper()) for p in prefixes):
                # 不是指定前缀开头的调用则跳过
                continue

        criteria.append(
            SlicingCriterion(
                node_id=node_id,
                kind="api_call",
                variable=callee_name,
                extra={"ast_type": cls_name},
            )
        )

    return criteria

# ======== 总入口：挖掘所有准则 =========

def mine_slicing_criteria(
    pdg: ProgramDependenceGraph,
    du: DefUseResult,
    symtab: POUSymbolTable,
    config: Optional[CriterionConfig] = None,
) -> List[SlicingCriterion]:
    """
    整体准则挖掘流程：

      - I/O 输出：io_output
      - 状态机：state_transition
      - 错误逻辑：error_logic
      - 状态机 / 控制逻辑：
          * stage_boundary
          * runtime_done / runtime_error / runtime_interrupt
      - 运动 / 物理量：
          * motion_quantity
          * motion_special_case
      - 关键 API 调用：按需扩展

    最后按 (node_id, kind, variable, access) 做一次去重。
    """
    if config is None:
        config = CriterionConfig()

    criteria: List[SlicingCriterion] = []

    # 1) 通用 I/O / 状态 / 错误
    criteria.extend(discover_io_output_criteria(symtab, pdg, du))
    state_vars = discover_state_variables(symtab, du, config)
    criteria.extend(discover_state_transition_criteria(state_vars, pdg, du))
    criteria.extend(discover_error_criteria(symtab, pdg, du, config))

    # 2) 状态机 / 控制逻辑
    criteria.extend(discover_stage_boundary_criteria(symtab, pdg, du, config))
    criteria.extend(discover_runtime_event_criteria(symtab, pdg, du, config))

     # 2.5) 纯语法控制结构：IF/CASE 分支 & 循环
    criteria.extend(discover_control_branch_criteria(pdg, du))
    criteria.extend(discover_loop_region_criteria(pdg, du))

    # 3) 运动 / 物理量
    criteria.extend(discover_motion_quantity_criteria(symtab, pdg, du, config))
    criteria.extend(discover_motion_special_case_criteria(symtab, pdg, du, config))

    # 4) 关键 API 调用（目前为空实现）
    criteria.extend(discover_api_call_criteria(pdg, du, config))

    # 5) 去重
    seen: Set[tuple] = set()
    deduped: List[SlicingCriterion] = []

    for c in criteria:
        extra = getattr(c, "extra", {}) or {}
        access = extra.get("access")
        if access is not None and hasattr(access, "pretty"):
            access_str = access.pretty()
        else:
            access_str = None

        key = (c.node_id, c.kind, c.variable, access_str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    return deduped
