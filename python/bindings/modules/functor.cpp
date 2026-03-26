/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/trampoline.h>

#include "../module.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"

namespace nb = nanobind;

namespace pypto {
namespace python {

using namespace pypto::ir;  // NOLINT(build/namespaces)

// Trampoline macros: generate NB_OVERRIDE_NAME + non-virtual base forwarder.
// The forwarder calls the base class method non-virtually, so that
// super().visit_*() in Python correctly calls the C++ default.

// Visitor trampolines (void return)
#define VISITOR_EXPR_TRAMPOLINE(CppType, py_name)                                                  \
  void VisitExpr_(const CppType##Ptr& op) override { NB_OVERRIDE_NAME(#py_name, VisitExpr_, op); } \
  void base_##py_name(const CppType##Ptr& op) { IRVisitor::VisitExpr_(op); }

#define VISITOR_STMT_TRAMPOLINE(CppType, py_name)                                                  \
  void VisitStmt_(const CppType##Ptr& op) override { NB_OVERRIDE_NAME(#py_name, VisitStmt_, op); } \
  void base_##py_name(const CppType##Ptr& op) { IRVisitor::VisitStmt_(op); }

// Mutator trampolines (ExprPtr/StmtPtr return)
#define MUTATOR_EXPR_TRAMPOLINE(CppType, py_name)                                                     \
  ExprPtr VisitExpr_(const CppType##Ptr& op) override { NB_OVERRIDE_NAME(#py_name, VisitExpr_, op); } \
  ExprPtr base_##py_name(const CppType##Ptr& op) { return IRMutator::VisitExpr_(op); }

#define MUTATOR_STMT_TRAMPOLINE(CppType, py_name)                                                     \
  StmtPtr VisitStmt_(const CppType##Ptr& op) override { NB_OVERRIDE_NAME(#py_name, VisitStmt_, op); } \
  StmtPtr base_##py_name(const CppType##Ptr& op) { return IRMutator::VisitStmt_(op); }

// --- IRVisitor trampoline ---
struct PyIRVisitor : IRVisitor {
  NB_TRAMPOLINE(IRVisitor, 55);  // 27 base + 23 binary + 5 unary

  // Top-level entry points
  void VisitProgram(const ProgramPtr& p) override { NB_OVERRIDE_NAME("visit_program", VisitProgram, p); }
  void base_visit_program(const ProgramPtr& p) { IRVisitor::VisitProgram(p); }

  void VisitFunction(const FunctionPtr& f) override { NB_OVERRIDE_NAME("visit_function", VisitFunction, f); }
  void base_visit_function(const FunctionPtr& f) { IRVisitor::VisitFunction(f); }

  // Dispatchers
  void VisitExpr(const ExprPtr& e) override { NB_OVERRIDE_NAME("visit_expr", VisitExpr, e); }
  void base_visit_expr(const ExprPtr& e) { IRVisitor::VisitExpr(e); }

  void VisitStmt(const StmtPtr& s) override { NB_OVERRIDE_NAME("visit_stmt", VisitStmt, s); }
  void base_visit_stmt(const StmtPtr& s) { IRVisitor::VisitStmt(s); }

  // Grouped handlers
  void VisitVarLike_(const VarPtr& op) override { NB_OVERRIDE_NAME("visit_var_like", VisitVarLike_, op); }
  void base_visit_var_like(const VarPtr& op) { IRVisitor::VisitVarLike_(op); }

  void VisitBinaryExpr_(const BinaryExprPtr& op) override {
    NB_OVERRIDE_NAME("visit_binary_expr", VisitBinaryExpr_, op);
  }
  void base_visit_binary_expr(const BinaryExprPtr& op) { IRVisitor::VisitBinaryExpr_(op); }

  void VisitUnaryExpr_(const UnaryExprPtr& op) override {
    NB_OVERRIDE_NAME("visit_unary_expr", VisitUnaryExpr_, op);
  }
  void base_visit_unary_expr(const UnaryExprPtr& op) { IRVisitor::VisitUnaryExpr_(op); }

  // Leaf expression types (9)
  VISITOR_EXPR_TRAMPOLINE(Var, visit_var)
  VISITOR_EXPR_TRAMPOLINE(IterArg, visit_iter_arg)
  VISITOR_EXPR_TRAMPOLINE(MemRef, visit_mem_ref)
  VISITOR_EXPR_TRAMPOLINE(ConstInt, visit_const_int)
  VISITOR_EXPR_TRAMPOLINE(ConstFloat, visit_const_float)
  VISITOR_EXPR_TRAMPOLINE(ConstBool, visit_const_bool)
  VISITOR_EXPR_TRAMPOLINE(Call, visit_call)
  VISITOR_EXPR_TRAMPOLINE(MakeTuple, visit_make_tuple)
  VISITOR_EXPR_TRAMPOLINE(TupleGetItemExpr, visit_tuple_get_item_expr)

  // Binary expression types (23) — individual overrides, default delegates to VisitBinaryExpr_
  VISITOR_EXPR_TRAMPOLINE(Add, visit_add)
  VISITOR_EXPR_TRAMPOLINE(Sub, visit_sub)
  VISITOR_EXPR_TRAMPOLINE(Mul, visit_mul)
  VISITOR_EXPR_TRAMPOLINE(FloorDiv, visit_floor_div)
  VISITOR_EXPR_TRAMPOLINE(FloorMod, visit_floor_mod)
  VISITOR_EXPR_TRAMPOLINE(FloatDiv, visit_float_div)
  VISITOR_EXPR_TRAMPOLINE(Min, visit_min)
  VISITOR_EXPR_TRAMPOLINE(Max, visit_max)
  VISITOR_EXPR_TRAMPOLINE(Pow, visit_pow)
  VISITOR_EXPR_TRAMPOLINE(Eq, visit_eq)
  VISITOR_EXPR_TRAMPOLINE(Ne, visit_ne)
  VISITOR_EXPR_TRAMPOLINE(Lt, visit_lt)
  VISITOR_EXPR_TRAMPOLINE(Le, visit_le)
  VISITOR_EXPR_TRAMPOLINE(Gt, visit_gt)
  VISITOR_EXPR_TRAMPOLINE(Ge, visit_ge)
  VISITOR_EXPR_TRAMPOLINE(And, visit_and)
  VISITOR_EXPR_TRAMPOLINE(Or, visit_or)
  VISITOR_EXPR_TRAMPOLINE(Xor, visit_xor)
  VISITOR_EXPR_TRAMPOLINE(BitAnd, visit_bit_and)
  VISITOR_EXPR_TRAMPOLINE(BitOr, visit_bit_or)
  VISITOR_EXPR_TRAMPOLINE(BitXor, visit_bit_xor)
  VISITOR_EXPR_TRAMPOLINE(BitShiftLeft, visit_bit_shift_left)
  VISITOR_EXPR_TRAMPOLINE(BitShiftRight, visit_bit_shift_right)

  // Unary expression types (5) — individual overrides, default delegates to VisitUnaryExpr_
  VISITOR_EXPR_TRAMPOLINE(Abs, visit_abs)
  VISITOR_EXPR_TRAMPOLINE(Neg, visit_neg)
  VISITOR_EXPR_TRAMPOLINE(Not, visit_not)
  VISITOR_EXPR_TRAMPOLINE(BitNot, visit_bit_not)
  VISITOR_EXPR_TRAMPOLINE(Cast, visit_cast)

  // Statement types (11 + generic fallback)
  VISITOR_STMT_TRAMPOLINE(AssignStmt, visit_assign_stmt)
  VISITOR_STMT_TRAMPOLINE(IfStmt, visit_if_stmt)
  VISITOR_STMT_TRAMPOLINE(ForStmt, visit_for_stmt)
  VISITOR_STMT_TRAMPOLINE(WhileStmt, visit_while_stmt)
  VISITOR_STMT_TRAMPOLINE(ScopeStmt, visit_scope_stmt)
  VISITOR_STMT_TRAMPOLINE(SeqStmts, visit_seq_stmts)
  VISITOR_STMT_TRAMPOLINE(YieldStmt, visit_yield_stmt)
  VISITOR_STMT_TRAMPOLINE(ReturnStmt, visit_return_stmt)
  VISITOR_STMT_TRAMPOLINE(EvalStmt, visit_eval_stmt)
  VISITOR_STMT_TRAMPOLINE(BreakStmt, visit_break_stmt)
  VISITOR_STMT_TRAMPOLINE(ContinueStmt, visit_continue_stmt)
};

// --- IRMutator trampoline ---
struct PyIRMutator : IRMutator {
  NB_TRAMPOLINE(IRMutator, 54);  // 26 base + 23 binary + 5 unary

  // Top-level entry points
  ProgramPtr VisitProgram(const ProgramPtr& p) override {
    NB_OVERRIDE_NAME("visit_program", VisitProgram, p);
  }
  ProgramPtr base_visit_program(const ProgramPtr& p) { return IRMutator::VisitProgram(p); }

  FunctionPtr VisitFunction(const FunctionPtr& f) override {
    NB_OVERRIDE_NAME("visit_function", VisitFunction, f);
  }
  FunctionPtr base_visit_function(const FunctionPtr& f) { return IRMutator::VisitFunction(f); }

  // Dispatchers
  ExprPtr VisitExpr(const ExprPtr& e) override { NB_OVERRIDE_NAME("visit_expr", VisitExpr, e); }
  ExprPtr base_visit_expr(const ExprPtr& e) { return IRMutator::VisitExpr(e); }

  StmtPtr VisitStmt(const StmtPtr& s) override { NB_OVERRIDE_NAME("visit_stmt", VisitStmt, s); }
  StmtPtr base_visit_stmt(const StmtPtr& s) { return IRMutator::VisitStmt(s); }

  // Grouped handlers
  ExprPtr VisitBinaryExpr_(const BinaryExprPtr& op) override {
    NB_OVERRIDE_NAME("visit_binary_expr", VisitBinaryExpr_, op);
  }
  ExprPtr base_visit_binary_expr(const BinaryExprPtr& op) { return IRMutator::VisitBinaryExpr_(op); }

  ExprPtr VisitUnaryExpr_(const UnaryExprPtr& op) override {
    NB_OVERRIDE_NAME("visit_unary_expr", VisitUnaryExpr_, op);
  }
  ExprPtr base_visit_unary_expr(const UnaryExprPtr& op) { return IRMutator::VisitUnaryExpr_(op); }

  // Leaf expression types (9)
  MUTATOR_EXPR_TRAMPOLINE(Var, visit_var)
  MUTATOR_EXPR_TRAMPOLINE(IterArg, visit_iter_arg)
  MUTATOR_EXPR_TRAMPOLINE(MemRef, visit_mem_ref)
  MUTATOR_EXPR_TRAMPOLINE(ConstInt, visit_const_int)
  MUTATOR_EXPR_TRAMPOLINE(ConstFloat, visit_const_float)
  MUTATOR_EXPR_TRAMPOLINE(ConstBool, visit_const_bool)
  MUTATOR_EXPR_TRAMPOLINE(Call, visit_call)
  MUTATOR_EXPR_TRAMPOLINE(MakeTuple, visit_make_tuple)
  MUTATOR_EXPR_TRAMPOLINE(TupleGetItemExpr, visit_tuple_get_item_expr)

  // Binary expression types (23)
  MUTATOR_EXPR_TRAMPOLINE(Add, visit_add)
  MUTATOR_EXPR_TRAMPOLINE(Sub, visit_sub)
  MUTATOR_EXPR_TRAMPOLINE(Mul, visit_mul)
  MUTATOR_EXPR_TRAMPOLINE(FloorDiv, visit_floor_div)
  MUTATOR_EXPR_TRAMPOLINE(FloorMod, visit_floor_mod)
  MUTATOR_EXPR_TRAMPOLINE(FloatDiv, visit_float_div)
  MUTATOR_EXPR_TRAMPOLINE(Min, visit_min)
  MUTATOR_EXPR_TRAMPOLINE(Max, visit_max)
  MUTATOR_EXPR_TRAMPOLINE(Pow, visit_pow)
  MUTATOR_EXPR_TRAMPOLINE(Eq, visit_eq)
  MUTATOR_EXPR_TRAMPOLINE(Ne, visit_ne)
  MUTATOR_EXPR_TRAMPOLINE(Lt, visit_lt)
  MUTATOR_EXPR_TRAMPOLINE(Le, visit_le)
  MUTATOR_EXPR_TRAMPOLINE(Gt, visit_gt)
  MUTATOR_EXPR_TRAMPOLINE(Ge, visit_ge)
  MUTATOR_EXPR_TRAMPOLINE(And, visit_and)
  MUTATOR_EXPR_TRAMPOLINE(Or, visit_or)
  MUTATOR_EXPR_TRAMPOLINE(Xor, visit_xor)
  MUTATOR_EXPR_TRAMPOLINE(BitAnd, visit_bit_and)
  MUTATOR_EXPR_TRAMPOLINE(BitOr, visit_bit_or)
  MUTATOR_EXPR_TRAMPOLINE(BitXor, visit_bit_xor)
  MUTATOR_EXPR_TRAMPOLINE(BitShiftLeft, visit_bit_shift_left)
  MUTATOR_EXPR_TRAMPOLINE(BitShiftRight, visit_bit_shift_right)

  // Unary expression types (5)
  MUTATOR_EXPR_TRAMPOLINE(Abs, visit_abs)
  MUTATOR_EXPR_TRAMPOLINE(Neg, visit_neg)
  MUTATOR_EXPR_TRAMPOLINE(Not, visit_not)
  MUTATOR_EXPR_TRAMPOLINE(BitNot, visit_bit_not)
  MUTATOR_EXPR_TRAMPOLINE(Cast, visit_cast)

  // Statement types (11 + generic fallback)
  MUTATOR_STMT_TRAMPOLINE(AssignStmt, visit_assign_stmt)
  MUTATOR_STMT_TRAMPOLINE(IfStmt, visit_if_stmt)
  MUTATOR_STMT_TRAMPOLINE(ForStmt, visit_for_stmt)
  MUTATOR_STMT_TRAMPOLINE(WhileStmt, visit_while_stmt)
  MUTATOR_STMT_TRAMPOLINE(ScopeStmt, visit_scope_stmt)
  MUTATOR_STMT_TRAMPOLINE(SeqStmts, visit_seq_stmts)
  MUTATOR_STMT_TRAMPOLINE(YieldStmt, visit_yield_stmt)
  MUTATOR_STMT_TRAMPOLINE(ReturnStmt, visit_return_stmt)
  MUTATOR_STMT_TRAMPOLINE(EvalStmt, visit_eval_stmt)
  MUTATOR_STMT_TRAMPOLINE(BreakStmt, visit_break_stmt)
  MUTATOR_STMT_TRAMPOLINE(ContinueStmt, visit_continue_stmt)
};

#undef VISITOR_EXPR_TRAMPOLINE
#undef VISITOR_STMT_TRAMPOLINE
#undef MUTATOR_EXPR_TRAMPOLINE
#undef MUTATOR_STMT_TRAMPOLINE

// Binding macros: bind each visit method via lambda to the base forwarder.
// Lambdas use the base class type (IRVisitor&/IRMutator&) to match nanobind's
// bound type. static_cast to the trampoline is safe because Python instances
// are always backed by PyIRVisitor/PyIRMutator.

#define BIND_VISITOR(cls, CppType, py_name)                                                                \
  cls.def(                                                                                                 \
      #py_name,                                                                                            \
      [](IRVisitor& self, const CppType##Ptr& op) { static_cast<PyIRVisitor&>(self).base_##py_name(op); }, \
      nb::arg("op"))

#define BIND_MUTATOR(cls, CppType, py_name)                        \
  cls.def(                                                         \
      #py_name,                                                    \
      [](IRMutator& self, const CppType##Ptr& op) {                \
        return static_cast<PyIRMutator&>(self).base_##py_name(op); \
      },                                                           \
      nb::arg("op"))

void BindFunctor(nb::module_& m) {
  nb::module_ ir_mod = nb::cast<nb::module_>(m.attr("ir"));

  // ---- IRVisitor ----
  auto visitor_cls =
      nb::class_<IRVisitor, PyIRVisitor>(
          ir_mod, "IRVisitor",
          "Read-only IR visitor. Subclass and override visit_* methods.\n\n"
          "Default implementations recursively traverse all children.\n"
          "Call super().visit_*() from your override to keep the default recursion.")
          .def(nb::init<>())
          .def(
              "visit_program", [](IRVisitor& self, const ProgramPtr& p) { self.IRVisitor::VisitProgram(p); },
              nb::arg("program"), "Visit all functions in a program")
          .def(
              "visit_function",
              [](IRVisitor& self, const FunctionPtr& f) { self.IRVisitor::VisitFunction(f); },
              nb::arg("func"), "Visit a function's parameters and body")
          .def(
              "visit_expr", [](IRVisitor& self, const ExprPtr& e) { self.IRVisitor::VisitExpr(e); },
              nb::arg("expr"), "Dispatch to type-specific expression handler")
          .def(
              "visit_stmt", [](IRVisitor& self, const StmtPtr& s) { self.IRVisitor::VisitStmt(s); },
              nb::arg("stmt"), "Dispatch to type-specific statement handler")
          .def(
              "visit_var_like",
              [](IRVisitor& self, const VarPtr& op) {
                static_cast<PyIRVisitor&>(self).base_visit_var_like(op);
              },
              nb::arg("op"), "Visit Var/IterArg shared logic (type shape expressions)")
          .def(
              "visit_binary_expr",
              [](IRVisitor& self, const BinaryExprPtr& op) {
                static_cast<PyIRVisitor&>(self).base_visit_binary_expr(op);
              },
              nb::arg("op"), "Visit any binary expression (default: visit left and right)")
          .def(
              "visit_unary_expr",
              [](IRVisitor& self, const UnaryExprPtr& op) {
                static_cast<PyIRVisitor&>(self).base_visit_unary_expr(op);
              },
              nb::arg("op"), "Visit any unary expression (default: visit operand)");

  // Leaf expression handlers
  BIND_VISITOR(visitor_cls, Var, visit_var);
  BIND_VISITOR(visitor_cls, IterArg, visit_iter_arg);
  BIND_VISITOR(visitor_cls, MemRef, visit_mem_ref);
  BIND_VISITOR(visitor_cls, ConstInt, visit_const_int);
  BIND_VISITOR(visitor_cls, ConstFloat, visit_const_float);
  BIND_VISITOR(visitor_cls, ConstBool, visit_const_bool);
  BIND_VISITOR(visitor_cls, Call, visit_call);
  BIND_VISITOR(visitor_cls, MakeTuple, visit_make_tuple);
  BIND_VISITOR(visitor_cls, TupleGetItemExpr, visit_tuple_get_item_expr);

  // Binary expression handlers (23)
  BIND_VISITOR(visitor_cls, Add, visit_add);
  BIND_VISITOR(visitor_cls, Sub, visit_sub);
  BIND_VISITOR(visitor_cls, Mul, visit_mul);
  BIND_VISITOR(visitor_cls, FloorDiv, visit_floor_div);
  BIND_VISITOR(visitor_cls, FloorMod, visit_floor_mod);
  BIND_VISITOR(visitor_cls, FloatDiv, visit_float_div);
  BIND_VISITOR(visitor_cls, Min, visit_min);
  BIND_VISITOR(visitor_cls, Max, visit_max);
  BIND_VISITOR(visitor_cls, Pow, visit_pow);
  BIND_VISITOR(visitor_cls, Eq, visit_eq);
  BIND_VISITOR(visitor_cls, Ne, visit_ne);
  BIND_VISITOR(visitor_cls, Lt, visit_lt);
  BIND_VISITOR(visitor_cls, Le, visit_le);
  BIND_VISITOR(visitor_cls, Gt, visit_gt);
  BIND_VISITOR(visitor_cls, Ge, visit_ge);
  BIND_VISITOR(visitor_cls, And, visit_and);
  BIND_VISITOR(visitor_cls, Or, visit_or);
  BIND_VISITOR(visitor_cls, Xor, visit_xor);
  BIND_VISITOR(visitor_cls, BitAnd, visit_bit_and);
  BIND_VISITOR(visitor_cls, BitOr, visit_bit_or);
  BIND_VISITOR(visitor_cls, BitXor, visit_bit_xor);
  BIND_VISITOR(visitor_cls, BitShiftLeft, visit_bit_shift_left);
  BIND_VISITOR(visitor_cls, BitShiftRight, visit_bit_shift_right);

  // Unary expression handlers (5)
  BIND_VISITOR(visitor_cls, Abs, visit_abs);
  BIND_VISITOR(visitor_cls, Neg, visit_neg);
  BIND_VISITOR(visitor_cls, Not, visit_not);
  BIND_VISITOR(visitor_cls, BitNot, visit_bit_not);
  BIND_VISITOR(visitor_cls, Cast, visit_cast);

  // Statement handlers
  BIND_VISITOR(visitor_cls, AssignStmt, visit_assign_stmt);
  BIND_VISITOR(visitor_cls, IfStmt, visit_if_stmt);
  BIND_VISITOR(visitor_cls, ForStmt, visit_for_stmt);
  BIND_VISITOR(visitor_cls, WhileStmt, visit_while_stmt);
  BIND_VISITOR(visitor_cls, ScopeStmt, visit_scope_stmt);
  BIND_VISITOR(visitor_cls, SeqStmts, visit_seq_stmts);
  BIND_VISITOR(visitor_cls, YieldStmt, visit_yield_stmt);
  BIND_VISITOR(visitor_cls, ReturnStmt, visit_return_stmt);
  BIND_VISITOR(visitor_cls, EvalStmt, visit_eval_stmt);
  BIND_VISITOR(visitor_cls, BreakStmt, visit_break_stmt);
  BIND_VISITOR(visitor_cls, ContinueStmt, visit_continue_stmt);

  // ---- IRMutator ----
  auto mutator_cls =
      nb::class_<IRMutator, PyIRMutator>(ir_mod, "IRMutator",
                                         "IR mutator with copy-on-write semantics.\n\n"
                                         "Subclass and override visit_* methods to transform IR.\n"
                                         "Default implementations recurse into children and reconstruct\n"
                                         "nodes only when children change (copy-on-write).")
          .def(nb::init<>())
          .def(
              "visit_program",
              [](IRMutator& self, const ProgramPtr& p) { return self.IRMutator::VisitProgram(p); },
              nb::arg("program"), "Mutate all functions in a program")
          .def(
              "visit_function",
              [](IRMutator& self, const FunctionPtr& f) { return self.IRMutator::VisitFunction(f); },
              nb::arg("func"), "Mutate a function's body")
          .def(
              "visit_expr", [](IRMutator& self, const ExprPtr& e) { return self.IRMutator::VisitExpr(e); },
              nb::arg("expr"), "Dispatch to type-specific expression mutator")
          .def(
              "visit_stmt", [](IRMutator& self, const StmtPtr& s) { return self.IRMutator::VisitStmt(s); },
              nb::arg("stmt"), "Dispatch to type-specific statement mutator")
          .def(
              "visit_binary_expr",
              [](IRMutator& self, const BinaryExprPtr& op) {
                return static_cast<PyIRMutator&>(self).base_visit_binary_expr(op);
              },
              nb::arg("op"), "Mutate any binary expression (default: visit children, reconstruct if changed)")
          .def(
              "visit_unary_expr",
              [](IRMutator& self, const UnaryExprPtr& op) {
                return static_cast<PyIRMutator&>(self).base_visit_unary_expr(op);
              },
              nb::arg("op"), "Mutate any unary expression (default: visit operand, reconstruct if changed)");

  // Leaf expression handlers
  BIND_MUTATOR(mutator_cls, Var, visit_var);
  BIND_MUTATOR(mutator_cls, IterArg, visit_iter_arg);
  BIND_MUTATOR(mutator_cls, MemRef, visit_mem_ref);
  BIND_MUTATOR(mutator_cls, ConstInt, visit_const_int);
  BIND_MUTATOR(mutator_cls, ConstFloat, visit_const_float);
  BIND_MUTATOR(mutator_cls, ConstBool, visit_const_bool);
  BIND_MUTATOR(mutator_cls, Call, visit_call);
  BIND_MUTATOR(mutator_cls, MakeTuple, visit_make_tuple);
  BIND_MUTATOR(mutator_cls, TupleGetItemExpr, visit_tuple_get_item_expr);

  // Binary expression handlers (23)
  BIND_MUTATOR(mutator_cls, Add, visit_add);
  BIND_MUTATOR(mutator_cls, Sub, visit_sub);
  BIND_MUTATOR(mutator_cls, Mul, visit_mul);
  BIND_MUTATOR(mutator_cls, FloorDiv, visit_floor_div);
  BIND_MUTATOR(mutator_cls, FloorMod, visit_floor_mod);
  BIND_MUTATOR(mutator_cls, FloatDiv, visit_float_div);
  BIND_MUTATOR(mutator_cls, Min, visit_min);
  BIND_MUTATOR(mutator_cls, Max, visit_max);
  BIND_MUTATOR(mutator_cls, Pow, visit_pow);
  BIND_MUTATOR(mutator_cls, Eq, visit_eq);
  BIND_MUTATOR(mutator_cls, Ne, visit_ne);
  BIND_MUTATOR(mutator_cls, Lt, visit_lt);
  BIND_MUTATOR(mutator_cls, Le, visit_le);
  BIND_MUTATOR(mutator_cls, Gt, visit_gt);
  BIND_MUTATOR(mutator_cls, Ge, visit_ge);
  BIND_MUTATOR(mutator_cls, And, visit_and);
  BIND_MUTATOR(mutator_cls, Or, visit_or);
  BIND_MUTATOR(mutator_cls, Xor, visit_xor);
  BIND_MUTATOR(mutator_cls, BitAnd, visit_bit_and);
  BIND_MUTATOR(mutator_cls, BitOr, visit_bit_or);
  BIND_MUTATOR(mutator_cls, BitXor, visit_bit_xor);
  BIND_MUTATOR(mutator_cls, BitShiftLeft, visit_bit_shift_left);
  BIND_MUTATOR(mutator_cls, BitShiftRight, visit_bit_shift_right);

  // Unary expression handlers (5)
  BIND_MUTATOR(mutator_cls, Abs, visit_abs);
  BIND_MUTATOR(mutator_cls, Neg, visit_neg);
  BIND_MUTATOR(mutator_cls, Not, visit_not);
  BIND_MUTATOR(mutator_cls, BitNot, visit_bit_not);
  BIND_MUTATOR(mutator_cls, Cast, visit_cast);

  // Statement handlers
  BIND_MUTATOR(mutator_cls, AssignStmt, visit_assign_stmt);
  BIND_MUTATOR(mutator_cls, IfStmt, visit_if_stmt);
  BIND_MUTATOR(mutator_cls, ForStmt, visit_for_stmt);
  BIND_MUTATOR(mutator_cls, WhileStmt, visit_while_stmt);
  BIND_MUTATOR(mutator_cls, ScopeStmt, visit_scope_stmt);
  BIND_MUTATOR(mutator_cls, SeqStmts, visit_seq_stmts);
  BIND_MUTATOR(mutator_cls, YieldStmt, visit_yield_stmt);
  BIND_MUTATOR(mutator_cls, ReturnStmt, visit_return_stmt);
  BIND_MUTATOR(mutator_cls, EvalStmt, visit_eval_stmt);
  BIND_MUTATOR(mutator_cls, BreakStmt, visit_break_stmt);
  BIND_MUTATOR(mutator_cls, ContinueStmt, visit_continue_stmt);
}

#undef BIND_VISITOR
#undef BIND_MUTATOR

}  // namespace python
}  // namespace pypto
