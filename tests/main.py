# tests/main.py

from pathlib import Path

from st_slicer.parser.parse_st import parse_st_code
from st_slicer.ast.builder import ASTBuilder
from st_slicer.ir.ir_builder import IRBuilder
from st_slicer.cfg.cfg_builder import CFGBuilder
from st_slicer.dataflow.def_use import DefUseAnalyzer
from st_slicer.pdg.pdg_builder import PDGBuilder, build_program_dependence_graph
from st_slicer.criteria import mine_slicing_criteria, CriterionConfig
from st_slicer.sema.builder import build_symbol_table

from st_slicer.functional_blocks import extract_functional_blocks
from st_slicer.block_context import build_completed_block


def main():
    # 这里改成你想测试的 ST 文件名
    code_path = Path(__file__).parent / "mc_moveabsolute.st"
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

        # 2) CFG
        cfg_builder = CFGBuilder(irb.instrs)
        cfg = cfg_builder.build()

        # 3) Def-Use（一定要把 ir2ast_stmt 传进去）
        du_analyzer = DefUseAnalyzer(cfg, ir2ast_stmt=irb.ir2ast_stmt)
        du_result = du_analyzer.analyze()

        # 4) PDG（后继风格）
        pdg_builder = PDGBuilder(cfg, du_result)
        raw_pdg = pdg_builder.build()

        # print("\n=== PDG Data Dependencies ===")
        # for src, dsts in sorted(raw_pdg.data_deps.items()):
        #     print(f"{src} --data--> {sorted(dsts)}")

        # print("\n=== PDG Control Dependencies ===")
        # for src, dsts in sorted(raw_pdg.control_deps.items()):
        #     print(f"{src} --ctrl--> {sorted(dsts)}")

        # 5) 构建“前驱风格”的 ProgramDependenceGraph（切片用）
        prog_pdg = build_program_dependence_graph(irb.instrs, raw_pdg)
        #print("\n=== ProgramDependenceGraph (predecessor view) ===")
        for nid, node in sorted(prog_pdg.nodes.items()):
            preds = prog_pdg.predecessors(nid)
            # if preds:
            #     print(f"node {nid} <- {preds}")

        # 6) 构符号表（project），再取当前 POU 的 symtab
        proj_symtab = build_symbol_table(pous)
        pou_symtab = proj_symtab.get_pou(pou.name)

        # print("\n=== Symbols in POU symtab ===")
        # for sym in pou_symtab.get_all_symbols():
        #     print(sym.name, getattr(sym, "type", None), getattr(sym, "role", None))

        # 7) 准则挖掘
        config = CriterionConfig()
        criteria = mine_slicing_criteria(prog_pdg, du_result, pou_symtab, config)

        print("\n=== Mined slicing criteria ===")
        for c in criteria:
            print(c)

        if not criteria:
            print("\nNo slicing criteria found for this POU.")
            continue

        # 8) 一键“多准则切片 + 聚类 + 块大小规范化”
        blocks = extract_functional_blocks(
            prog_pdg=prog_pdg,
            criteria=criteria,
            ir2ast_stmt=irb.ir2ast_stmt,
            code_lines=code_lines,
            overlap_threshold=0.5,  # 两个切片重叠比例 >= 0.5 就归为同一功能块
            min_lines=20,           # 每个块至少 20 行
            max_lines=150,          # 每个块最多 150 行
        )

        print(f"\nTotal functional blocks (after size normalization): {len(blocks)}")

        # 9) 对每个功能块做结构补全，生成独立 PROGRAM
        for idx, block in enumerate(blocks):
            print(f"\n\n===== Functional Block #{idx} =====")
            print(f"Lines in this block: {len(block.line_numbers)} -> {sorted(block.line_numbers)[:10]} ...")
            print(f"Nodes in this block: {len(block.node_ids)}")

            print("\n--- ST code for this functional block (original snippet) ---")
            for ln in sorted(block.line_numbers):
                print(f"{ln:4d}: {code_lines[ln-1].rstrip()}")

            # 生成完整 PROGRAM（带 VAR 区）
            completed = build_completed_block(
                block=block,
                pou_name=pou.name,
                pou_symtab=pou_symtab,
                code_lines=code_lines,
                block_index=idx,
            )

            print("\n--- Completed ST program for this functional block ---")
            print(completed.code)

            # 如果你希望写成单独文件，可以取消下面注释：
            # out_name = f"{pou.name}_block_{idx}.st"
            # out_path = code_path.parent / out_name
            # out_path.write_text(completed.code, encoding="utf-8")
            # print(f"\n[Saved completed block to {out_path}]")


if __name__ == "__main__":
    main()
