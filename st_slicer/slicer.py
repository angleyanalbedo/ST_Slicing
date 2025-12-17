# st_slicer/slicer.py

from dataclasses import dataclass, field
from typing import Iterable, Set, Literal, Optional, Dict

from .pdg.pdg_builder import ProgramDependenceGraph
from blocks.types import SlicingCriterion

def backward_slice(
    pdg: ProgramDependenceGraph,
    start_nodes: Iterable[int],
    use_data: bool = True,
    use_control: bool = True,
) -> Set[int]:
    """
    基于 PDG 的静态后向切片，考虑 data + control 两类前驱。
    """
    worklist = list(start_nodes)
    slice_nodes: Set[int] = set()

    while worklist:
        n = worklist.pop()
        if n in slice_nodes:
            continue
        slice_nodes.add(n)

        for pred, edge_type in pdg.predecessors(n):
            if (edge_type == "data" and not use_data) or \
               (edge_type == "control" and not use_control):
                continue
            # 目前 data / control 全都要；如果之后想做变量敏感可以在这里收紧
            if pred not in slice_nodes:
                worklist.append(pred)

    return slice_nodes
