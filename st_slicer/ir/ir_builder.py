# ir/ir_builder.py
from __future__ import annotations

from typing import List, Optional, Tuple, Dict
from ..ast import nodes as astn
from ..ast.nodes import (
    SourceLocation,
    Stmt,
    Expr,
    Assignment,
    IfStmt,
    ForStmt,
    CallStmt,
    VarRef,
    Literal,
    BinOp,
)
from .ir_nodes import (
    IRInstr,
    IRLocation,
    IRAssign,
    IRBinOp,
    IRCall,
    IRBranchCond,
    IRLabel,
    IRGoto,
)


class IRBuilder:
    def __init__(self, pou_name: str):
        self.pou_name = pou_name

        # 统一用 instrs 这个名字
        self.instrs: List[IRInstr] = []

        # IR index -> AST 语句
        self.ir2ast_stmt: List[Optional[Stmt]] = []

        # AST 语句 -> IR indices
        #self.ast2ir_indices: Dict[Stmt, List[int]] = {}

        self.temp_id: int = 0
        self.label_id: int = 0

    # 统一的 emit 出口
    def emit(self, instr: IRInstr, ast_stmt: Optional[Stmt] = None) -> int:
        idx = len(self.instrs)
        # 可选：把 ast_stmt 挂在 instr 上，便于调试
        setattr(instr, "ast_stmt", ast_stmt)

        self.instrs.append(instr)
        self.ir2ast_stmt.append(ast_stmt)

        # if ast_stmt is not None:
        #     self.ast2ir_indices.setdefault(ast_stmt, []).append(idx)

        return idx

    def new_temp(self) -> str:
        self.temp_id += 1
        return f"t{self.temp_id}"

    def new_label(self, prefix: str) -> str:
        self.label_id += 1
        return f"{prefix}_{self.label_id}"

    def _loc(self, ast_node) -> IRLocation:
        loc: SourceLocation = ast_node.loc
        return IRLocation(
            pou=self.pou_name,
            file=loc.file,
            line=loc.line,
        )

    # ========= 表达式 =========

    def lower_expr(self, expr: Expr) -> str:
        if isinstance(expr, VarRef):
            return expr.name

        if isinstance(expr, Literal):
            t = self.new_temp()
            self.emit(
                IRAssign(
                    target=t,
                    src=str(expr.value),
                    loc=self._loc(expr),
                ),
                ast_stmt=None,   # 表达式级 IR，不绑定到语句
            )
            return t

        if isinstance(expr, BinOp):
            left = self.lower_expr(expr.left)
            right = self.lower_expr(expr.right)
            t = self.new_temp()
            self.emit(
                IRBinOp(
                    dest=t,
                    op=expr.op,
                    left=left,
                    right=right,
                    loc=self._loc(expr),
                ),
                ast_stmt=None,
            )
            return t

        raise NotImplementedError(f"lower_expr not implemented for {type(expr)}")

    def _lower_lvalue(self, expr: Expr) -> str:
        if isinstance(expr, VarRef):
            return expr.name
        raise NotImplementedError(f"lvalue not supported for {type(expr)}")

    # ========= 语句入口 =========

    def lower_stmt(self, stmt: Stmt):
        if isinstance(stmt, Assignment):
            self._lower_assignment(stmt)
        elif isinstance(stmt, IfStmt):
            self._lower_if(stmt)
        elif isinstance(stmt, ForStmt):
            self._lower_for(stmt)
        elif isinstance(stmt, CallStmt):
            self._lower_call(stmt)
        else:
            # 其他语句先忽略
            pass

    # ========= 各类语句 =========

    def _lower_assignment(self, stmt: Assignment):
        rhs = self.lower_expr(stmt.value)
        target = self._lower_lvalue(stmt.target)
        self.emit(
            IRAssign(
                target=target,
                src=rhs,
                loc=self._loc(stmt),
            ),
            ast_stmt=stmt,   # 关键：绑定到这条 Assignment 语句
        )

    def _lower_call(self, stmt: CallStmt):
        arg_vars: List[str] = [self.lower_expr(a) for a in stmt.args]
        self.emit(
            IRCall(
                dest=None,
                callee=stmt.fb_name,
                args=arg_vars,
                loc=self._loc(stmt),
            ),
            ast_stmt=stmt,
        )

    def _lower_if(self, stmt: IfStmt):
        branches: List[Tuple[Expr, List[Stmt]]] = []
        branches.append((stmt.cond, stmt.then_body))
        elif_list: List[Tuple[Expr, List[Stmt]]] = getattr(stmt, "elif_branches", [])
        branches.extend(elif_list)

        has_else = bool(stmt.else_body)
        label_end = self.new_label("endif")
        label_else: Optional[str] = self.new_label("else") if has_else else None

        for idx, (cond_expr, then_body) in enumerate(branches):
            is_last = (idx == len(branches) - 1)
            cond_var = self.lower_expr(cond_expr)
            label_then = self.new_label(f"then_{idx}")

            if is_last:
                label_false = label_else if has_else and label_else is not None else label_end
            else:
                label_false = self.new_label(f"if_next_{idx}")

            self.emit(
                IRBranchCond(
                    cond=cond_var,
                    true_label=label_then,
                    false_label=label_false,
                    loc=self._loc(cond_expr),
                ),
                ast_stmt=stmt,
            )

            self.emit(
                IRLabel(
                    name=label_then,
                    loc=self._loc(stmt),
                ),
                ast_stmt=stmt,
            )
            for s in then_body:
                self.lower_stmt(s)
            self.emit(
                IRGoto(
                    target_label=label_end,
                    loc=self._loc(stmt),
                ),
                ast_stmt=stmt,
            )

            if not is_last:
                self.emit(
                    IRLabel(
                        name=label_false,
                        loc=self._loc(stmt),
                    ),
                    ast_stmt=stmt,
                )

        if has_else and label_else is not None:
            self.emit(
                IRLabel(
                    name=label_else,
                    loc=self._loc(stmt),
                ),
                ast_stmt=stmt,
            )
            for s in stmt.else_body:
                self.lower_stmt(s)
            self.emit(
                IRGoto(
                    target_label=label_end,
                    loc=self._loc(stmt),
                ),
                ast_stmt=stmt,
            )

        self.emit(
            IRLabel(
                name=label_end,
                loc=self._loc(stmt),
            ),
            ast_stmt=stmt,
        )

    def _lower_for(self, stmt: ForStmt):
        loop_var_name: str = stmt.var

        start_val = self.lower_expr(stmt.start)
        self.emit(
            IRAssign(
                target=loop_var_name,
                src=start_val,
                loc=self._loc(stmt),
            ),
            ast_stmt=stmt,
        )

        label_header = self.new_label("for_header")
        label_body = self.new_label("for_body")
        label_end = self.new_label("for_end")

        self.emit(IRLabel(name=label_header, loc=self._loc(stmt)), ast_stmt=stmt)

        end_val = self.lower_expr(stmt.end)
        t_cond = self.new_temp()
        self.emit(
            IRBinOp(
                dest=t_cond,
                op="<=",
                left=loop_var_name,
                right=end_val,
                loc=self._loc(stmt),
            ),
            ast_stmt=stmt,
        )

        self.emit(
            IRBranchCond(
                cond=t_cond,
                true_label=label_body,
                false_label=label_end,
                loc=self._loc(stmt),
            ),
            ast_stmt=stmt,
        )

        self.emit(IRLabel(name=label_body, loc=self._loc(stmt)), ast_stmt=stmt)
        for s in stmt.body:
            self.lower_stmt(s)

        if stmt.step is not None:
            step_val = self.lower_expr(stmt.step)
        else:
            t_step = self.new_temp()
            self.emit(
                IRAssign(
                    target=t_step,
                    src="1",
                    loc=self._loc(stmt),
                ),
                ast_stmt=stmt,
            )
            step_val = t_step

        t_next = self.new_temp()
        self.emit(
            IRBinOp(
                dest=t_next,
                op="+",
                left=loop_var_name,
                right=step_val,
                loc=self._loc(stmt),
            ),
            ast_stmt=stmt,
        )
        self.emit(
            IRAssign(
                target=loop_var_name,
                src=t_next,
                loc=self._loc(stmt),
            ),
            ast_stmt=stmt,
        )

        self.emit(
            IRGoto(
                target_label=label_header,
                loc=self._loc(stmt),
            ),
            ast_stmt=stmt,
        )
        self.emit(
            IRLabel(
                name=label_end,
                loc=self._loc(stmt),
            ),
            ast_stmt=stmt,
        )
