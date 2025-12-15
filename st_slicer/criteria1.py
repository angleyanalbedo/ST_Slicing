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

    # ↓↓↓ 新增：运动控制领域相关的名字启发 ↓↓↓
    motion_axis_name_keywords: Optional[List[str]] = None
    motion_profile_name_keywords: Optional[List[str]] = None
    motion_stage_name_candidates: Optional[List[str]] = None

    def __post_init__(self):
        if self.error_name_keywords is None:
            self.error_name_keywords = ["error", "err", "alarm", "diag", "status", "fault"]
        if self.state_name_keywords is None:
            self.state_name_keywords = ["stage", "state", "mode", "step", "phase"]
        if self.api_name_prefixes is None:
            # 后续可以根据项目再加，比如 ["MC_", "TCP_", "LOG_"]
            self.api_name_prefixes = ["MC_"]

        # ↓↓↓ 运动轴变量：Axis_state / Axis_maxvel / Axis_taget_postion 等 ↓↓↓
        if self.motion_axis_name_keywords is None:
            # 这里只匹配名字里包含 "axis_" 的变量
            self.motion_axis_name_keywords = ["axis_"]

        # ↓↓↓ 运动学轮廓参数：速度 / 加速度 / 距离 / 时间等 ↓↓↓
        if self.motion_profile_name_keywords is None:
            # 结合你 MC_MoveAbsolute 里实际出现的变量来写：
            self.motion_profile_name_keywords = [
                "vel",      # Axis_maxvel, Axis_now_velocity, ...
                "acc",      # Axis_maxacc, s_acc, ...
                "dec",      # Axis_maxdec, s_dec, ...
                "jerk",     # Axis_maxjerk
                "s_all", "t_all",
                "s_acc", "s_dec", "s_blend",
                "s_3seg", "t_3seg",
                "t_now",
            ]

        # ↓↓↓ 状态机里的主 stage 变量（主要针对 MC_* 程序里的 stage）↓↓↓
        if self.motion_stage_name_candidates is None:
            # 大部分 MC 程序里就是 "stage"；如果有 "stageX" 等也能命中
            self.motion_stage_name_candidates = ["stage"]


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

# ============ 2.x 运动控制：阶段边界准则 ============

def discover_motion_stage_boundary_criteria(
    symtab: POUSymbolTable,
    pdg: ProgramDependenceGraph,
    du: DefUseResult,
    config: CriterionConfig,
) -> List[SlicingCriterion]:
    """
    领域化的阶段边界准则：

    在 MC_* 程序里，典型的阶段切换语句会同时：
      - 给 stage/state 这种状态变量赋值；
      - 给 Busy / Active / Done / Error 这类输出变量赋值。

    我们把这类 “stage + 输出一起写” 的节点提取出来，
    用 kind="motion_stage_boundary" 标记为切片准则。
    """

    criteria: List[SlicingCriterion] = []

    # 1) 找出所有状态变量（复用你已有的 discover_state_variables）
    state_vars = discover_state_variables(symtab, du, config)
    if not state_vars:
        return criteria

    # 进一步筛出“更像运动阶段”的变量（名字里包含 "stage"）
    motion_stage_vars: Set[str] = set(
        v
        for v in state_vars
        if _match_any_keyword(v, config.motion_stage_name_candidates or [])
    )
    if not motion_stage_vars:
        # 没找到专门的 stage 变量，就退而求其次用所有状态变量
        motion_stage_vars = set(state_vars)

    # 2) 找出所有输出变量（复用 I/O 输出准则里的识别逻辑）
    output_bases: Set[str] = set()
    for name, sym in symtab.vars.items():
        storage = getattr(sym, "storage", "") or ""
        s_up = storage.upper()
        if ("OUTPUT" in s_up) or ("IN_OUT" in s_up):
            output_bases.add(name)

    if not output_bases:
        return criteria

    # 3) 在 PDG 节点上同时写 stage 和输出变量的点视为“阶段边界”
    for node_id in pdg.nodes.keys():
        if node_id < 0 or node_id >= len(du.def_vars):
            continue

        def_names = du.def_vars[node_id]
        has_stage = any(v in motion_stage_vars for v in def_names)
        has_output = any(v in output_bases for v in def_names)

        # 再看结构化 VarAccess（数组/结构）里的 base
        acc = _collect_node_var_accesses(node_id, du)
        bases = {va.base for va in acc["def"]}

        if not has_stage:
            has_stage = any(b in motion_stage_vars for b in bases)
        if not has_output:
            has_output = any(b in output_bases for b in bases)

        if not (has_stage and has_output):
            continue

        # 选一个最典型的 stage 变量名填到 variable 字段
        stage_name = next((v for v in def_names if v in motion_stage_vars), None)
        if stage_name is None:
            for va in acc["def"]:
                if va.base in motion_stage_vars:
                    stage_name = va.base
                    break

        criteria.append(
            SlicingCriterion(
                node_id=node_id,
                kind="motion_stage_boundary",
                variable=stage_name or "<stage>",
                extra={
                    "stage_vars": sorted(motion_stage_vars),
                    "output_bases": sorted(output_bases),
                },
            )
        )

    return criteria

# ============ 3. 运动控制：轴参数准则 ============

def discover_motion_axis_param_criteria(
    symtab: POUSymbolTable,
    pdg: ProgramDependenceGraph,
    du: DefUseResult,
    config: CriterionConfig,
) -> List[SlicingCriterion]:
    """
    轴参数准则：

    变量名里带 "Axis_" 的通常是：
      - 轴状态 / 轴模式 (Axis_state, Axis_mode)
      - 轴速度 / 加速度上限 (Axis_maxvel, Axis_maxacc, ...)
      - 轴目标位置 / 相对位移 (Axis_taget_postion, Axis_relative_postion)
      - 轴当前插补速度/位置 (Axis_now_motion_vel, Axis_now_motion_aimpos, ...)

    这些变量的定义点会生成 kind="axis_param" 的切片准则。
    """

    criteria: List[SlicingCriterion] = []

    axis_keywords = config.motion_axis_name_keywords or []
    if not axis_keywords:
        return criteria

    for node_id in pdg.nodes.keys():
        if node_id < 0 or node_id >= len(du.def_vars):
            continue

        # 1) 标量变量的 def
        for v in du.def_vars[node_id]:
            if _match_any_keyword(v, axis_keywords):
                criteria.append(
                    SlicingCriterion(
                        node_id=node_id,
                        kind="axis_param",
                        variable=v,
                        extra={
                            "base": v,
                            "access": None,
                        },
                    )
                )

        # 2) 结构化 VarAccess 的 def（例如 Axis_array[i]）
        acc = _collect_node_var_accesses(node_id, du)
        for va in acc["def"]:
            if _match_any_keyword(va.base, axis_keywords):
                criteria.append(
                    SlicingCriterion(
                        node_id=node_id,
                        kind="axis_param",
                        variable=va.pretty(),
                        extra={
                            "base": va.base,
                            "access": va,
                        },
                    )
                )

    return criteria


# ============ 4. 运动控制：运动轮廓参数准则 ============

def discover_motion_profile_criteria(
    symtab: POUSymbolTable,
    pdg: ProgramDependenceGraph,
    du: DefUseResult,
    config: CriterionConfig,
) -> List[SlicingCriterion]:
    """
    运动轮廓准则：

    变量名里包含 vel/acc/dec/jerk 以及 s_all/t_all/s_acc/s_dec/s_blend/t_now 等，
    视为“运动学轮廓参数”。它们的定义点会生成 kind="motion_profile" 的准则。

    典型例子（MC_MoveAbsolute）：
      - 计算 s_all / s_acc / s_dec / s_blend / t_all / t_3seg / s_3seg
      - 计算 vel (vs, vf, Axis_now_motion_vel, ...)
    """

    criteria: List[SlicingCriterion] = []

    profile_keywords = config.motion_profile_name_keywords or []
    if not profile_keywords:
        return criteria

    for node_id in pdg.nodes.keys():
        if node_id < 0 or node_id >= len(du.def_vars):
            continue

        # 1) 标量变量的 def
        for v in du.def_vars[node_id]:
            if _match_any_keyword(v, profile_keywords):
                criteria.append(
                    SlicingCriterion(
                        node_id=node_id,
                        kind="motion_profile",
                        variable=v,
                        extra={
                            "base": v,
                            "access": None,
                        },
                    )
                )

        # 2) 结构化 VarAccess 的 def
        acc = _collect_node_var_accesses(node_id, du)
        for va in acc["def"]:
            if _match_any_keyword(va.base, profile_keywords):
                criteria.append(
                    SlicingCriterion(
                        node_id=node_id,
                        kind="motion_profile",
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
      - 状态机：状态变量的定义点
      - 错误/报警：Error/ErrorID 等变量的定义点
      - 运动控制（新增）：
          * 阶段边界：stage + 输出一起赋值的节点
          * 轴参数：Axis_* 相关变量的定义点
          * 运动轮廓：vel/acc/dec/jerk/s_all/t_all/... 等轮廓变量的定义点
      - 关键 API 调用：暂留，可用于 MC_WrAxisPar_* / MC_Move* 调用
    """
    if config is None:
        config = CriterionConfig()

    criteria: List[SlicingCriterion] = []

    # 1. I/O 输出
    criteria.extend(discover_io_output_criteria(symtab, pdg, du))

    # 2. 状态机
    state_vars = discover_state_variables(symtab, du, config)
    criteria.extend(discover_state_transition_criteria(state_vars, pdg, du))

    # 2.x 运动控制：阶段边界
    criteria.extend(discover_motion_stage_boundary_criteria(symtab, pdg, du, config))

    # 3. 错误/报警
    criteria.extend(discover_error_criteria(symtab, pdg, du, config))

    # 3.x 运动控制：轴参数 & 运动轮廓
    criteria.extend(discover_motion_axis_param_criteria(symtab, pdg, du, config))
    criteria.extend(discover_motion_profile_criteria(symtab, pdg, du, config))

    # 4. 关键 API 调用（目前为空实现）
    criteria.extend(discover_api_call_criteria(pdg, config))

    # 5. 简单去重：按 (node_id, kind, base/variable, access) 归一
    deduped: List[SlicingCriterion] = []
    seen: Set[tuple] = set()

    for c in criteria:
        extra = getattr(c, "extra", {}) or {}
        access = extra.get("access", None)
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
