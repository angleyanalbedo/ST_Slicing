# st_slicer/tests/main.py

from pathlib import Path

from st_slicer.parser.parse_st import parse_st_code
from st_slicer.ast.builder import ASTBuilder
from st_slicer.ir.ir_builder import IRBuilder
from st_slicer.cfg.cfg_builder import CFGBuilder
from st_slicer.dataflow.def_use import DefUseAnalyzer
from st_slicer.pdg.pdg_builder import PDGBuilder, build_program_dependence_graph
from st_slicer.criteria import mine_slicing_criteria, CriterionConfig
from st_slicer.slicer import backward_slice
from st_slicer.sema.builder import build_symbol_table



def main():
    # 你可以在这里随时换测试文件，如 demo_array.st / demo_struct.st
    code_path = Path(__file__).parent / "19.st"
    code = code_path.read_text(encoding="utf-8")
    code_lines = code.splitlines()

    tree = parse_st_code(code)
    builder = ASTBuilder(filename=str(code_path))
    pous = builder.visit(tree)

    print("POU 数量:", len(pous))
    for pou in pous:
        print("POU:", pou.name, "vars:", len(pou.vars), "stmts:", len(pou.body))

        # 1) IR
        irb = IRBuilder(pou_name=pou.name)
        for s in pou.body:
            irb.lower_stmt(s)

        # print(f"\n=== IR for === {pou.name} ===")
        # for i, ins in enumerate(irb.instrs):
        #     print(i, type(ins).__name__, vars(ins))

        # print("\n=== IR -> AST stmt mapping ===")
        # for i, ins in enumerate(irb.instrs):
        #     ast_stmt = irb.ir2ast_stmt[i] if i < len(irb.ir2ast_stmt) else None
        #     if ast_stmt is None:
        #         ast_info = "None (expr-level IR / label/goto etc.)"
        #     else:
        #         ast_info = f"{type(ast_stmt).__name__} @ line {getattr(ast_stmt.loc, 'line', '?')}"
        #     print(f"{i}: {ast_info}")

        # 2) CFG
        cfg_builder = CFGBuilder(irb.instrs)
        cfg = cfg_builder.build()

        # print("\n=== CFG succ ===")
        # for i in range(len(irb.instrs)):
        #     print(f"{i} -> {cfg.succ[i]}")

        # 3) Def-Use（一定要把 ir2ast_stmt 传进去）
        du_analyzer = DefUseAnalyzer(cfg, ir2ast_stmt=irb.ir2ast_stmt)
        du_result = du_analyzer.analyze()

        # print("\n=== DEF/USE per instruction (VarName) ===")
        # for i in range(len(irb.instrs)):
        #     print(f"{i}: DEF={du_result.def_vars[i]} USE={du_result.use_vars[i]}")

        # print("\n=== Structured DEF/USE per instruction (VarAccess.pretty) ===")
        # for i in range(len(irb.instrs)):
        #     def_str = {va.pretty() for va in getattr(du_result, "def_accesses", [set()])[i]}
        #     use_str = {va.pretty() for va in getattr(du_result, "use_accesses", [set()])[i]}
        #     print(f"{i}: DEF={def_str} USE={use_str}")

        # 4) PDG（后继风格）
        pdg_builder = PDGBuilder(cfg, du_result)
        raw_pdg = pdg_builder.build()

        print("\n=== PDG Data Dependencies ===")
        for src, dsts in sorted(raw_pdg.data_deps.items()):
            print(f"{src} --data--> {sorted(dsts)}")

        print("\n=== PDG Control Dependencies ===")
        for src, dsts in sorted(raw_pdg.control_deps.items()):
            print(f"{src} --ctrl--> {sorted(dsts)}")

        # 5) 构建“前驱风格”的 ProgramDependenceGraph（切片用）
        prog_pdg = build_program_dependence_graph(irb.instrs, raw_pdg)
        print("\n=== ProgramDependenceGraph (predecessor view) ===")
        for nid, node in sorted(prog_pdg.nodes.items()):
            preds = prog_pdg.predecessors(nid)
            if preds:
                print(f"node {nid} <- {preds}")

        # 6) 构符号表（project），再取当前 POU 的 symtab
        proj_symtab = build_symbol_table(pous)
        pou_symtab = proj_symtab.get_pou(pou.name)   # 具体接口按你的 symbols 实现

        print("\n=== Symbols in POU symtab ===")
        for sym in pou_symtab.get_all_symbols():
            print(sym.name, getattr(sym, "type", None), getattr(sym, "role", None))


        # 7) 准则挖掘
        config = CriterionConfig()
        criteria = mine_slicing_criteria(prog_pdg, du_result, pou_symtab, config)

        print("\n=== Mined slicing criteria ===")
        for c in criteria:
            print(c)

        # 3) 如果没有准则，就没法切片，直接跳过后续映射
        if not criteria:
            print("\nNo slicing criteria found for this POU.")
            continue

        # 4) 随便拿一个准则做切片（这里取第 0 个）
        crit = criteria[5]
        print(f"\n=== Run slicing for first criterion: node={crit.node_id}, kind={crit.kind}, var={crit.variable} ===")
        slice_nodes = backward_slice(prog_pdg, [crit.node_id])
        print("Slice nodes for first criterion:", sorted(slice_nodes))

        # 5) 把 IR 指令切片映射回“语句级 + 源码级”
        # (1) IR -> AST 语句去重
        stmt_set = set()
        for i in slice_nodes:
            if 0 <= i < len(irb.ir2ast_stmt):
                ast_stmt = irb.ir2ast_stmt[i]
                if ast_stmt is not None:
                    stmt_set.add(ast_stmt)

        # (2) AST 语句 -> 源码行号
        sliced_lines = set()
        for st in stmt_set:
            line_no = getattr(st.loc, "line", None)
            if line_no is not None and 1 <= line_no <= len(code_lines):
                sliced_lines.add(line_no)

        print("Sliced source lines:", sorted(sliced_lines))
        print("\n=== Sliced ST code (by line) ===")
        for ln in sorted(sliced_lines):
            print(f"{ln:4d}: {code_lines[ln-1].rstrip()}")

        # 之后你可以进一步：基于 AST 构造一个新的 Program AST，只保留这些语句，再 pretty-print 为完整可编译的 ST 程序。



if __name__ == "__main__":
    main()

