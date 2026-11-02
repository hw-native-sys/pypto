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

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <type_traits>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace {

using namespace pypto;      // NOLINT(build/namespaces)
using namespace pypto::ir;  // NOLINT(build/namespaces)

// Exercise raw enum values here: Python enum conversion cannot express every
// malformed native caller, but the public C++ constructors must reject them too.
template <typename Callback>
void ExpectValueError(Callback callback) {
  bool caught = false;
  try {
    callback();
  } catch (const ValueError&) {
    caught = true;
  }
  assert(caught);
}

void CheckRegisterContracts(uint8_t invalid_kind, uint8_t invalid_wide_kind) {
  static_assert(std::is_base_of_v<Type, RegType>);
  static_assert(!std::is_base_of_v<ShapedType, RegType>);
  RegTypePtr vec = std::make_shared<RegType>(64, DataType::FP32, RegKind::VecReg);
  RegTypePtr mask = std::make_shared<RegType>(64, DataType::BOOL, RegKind::MaskReg);
  assert(vec->GetShape() == 64 && vec->GetLanes() == 64);
  assert(vec->GetDType() == DataType::FP32 && vec->GetRegKind() == RegKind::VecReg);
  assert(IsVecRegType(vec) && !IsMaskRegType(vec));
  assert(IsMaskRegType(mask) && !IsVecRegType(mask));
  TypePtr type = vec;
  assert(IsA<Type>(type) && !IsA<ShapedType>(type));
  ExpectValueError([] { RegType(0, DataType::FP32, RegKind::VecReg); });
  ExpectValueError([] { RegType(-1, DataType::FP32, RegKind::VecReg); });
  ExpectValueError([] { RegType(64, DataType::BOOL, RegKind::VecReg); });
  ExpectValueError([] { RegType(64, DataType::FP32, RegKind::MaskReg); });
  ExpectValueError([&] { RegType(64, DataType::FP32, static_cast<RegKind>(invalid_kind)); });
  ExpectValueError([&] { RegType(64, DataType::FP32, static_cast<RegKind>(invalid_wide_kind)); });
  ExpectValueError([] { RegType(64, DataType(255), RegKind::VecReg); });
}

void CheckStatementContracts(uint8_t invalid_policy) {
  auto span = Span::unknown();
  auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span);
  RegScopeStmtPtr scope = std::make_shared<RegScopeStmt>("vf", body, span);
  StmtPtr stmt = scope;
  assert(IsA<ScopeStmt>(stmt) && IsA<Stmt>(stmt));
  assert(scope->GetScopeKind() == ScopeKind::Reg);
  assert(static_cast<uint8_t>(ScopeKind::Reg) == 9);
  auto index = std::make_shared<Var>("i", std::make_shared<ScalarType>(DataType::INDEX), span);
  auto zero = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  auto one = std::make_shared<ConstInt>(1, DataType::INDEX, span);
  ForStmt automatic(index, zero, one, one, {}, body, {}, span);
  assert(automatic.GetHardwareLoopPolicy() == HardwareLoopPolicy::Auto);
  ForStmt required(index, zero, one, one, {}, body, {}, span, ForKind::Sequential, {}, {},
                   HardwareLoopPolicy::Require);
  assert(required.GetHardwareLoopPolicy() == HardwareLoopPolicy::Require);
  ExpectValueError([&] {
    ForStmt(index, zero, one, one, {}, body, {}, span, ForKind::Sequential, {}, {},
            static_cast<HardwareLoopPolicy>(invalid_policy));
  });
}

}  // namespace

int main(int argc, char* argv[]) {
  assert(argc == 4);
  // CTest supplies malformed caller values at runtime, so the analyzer checks
  // constructor validation rather than reporting deliberate constant enum casts.
  auto invalid_kind = static_cast<uint8_t>(std::strtoul(argv[1], nullptr, 10));
  auto invalid_wide_kind = static_cast<uint8_t>(std::strtoul(argv[2], nullptr, 10));
  auto invalid_policy = static_cast<uint8_t>(std::strtoul(argv[3], nullptr, 10));
  CheckRegisterContracts(invalid_kind, invalid_wide_kind);
  CheckStatementContracts(invalid_policy);
}
