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

#include <any>
#include <cctype>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

bool IsSupportedPrintfConversion(char conversion) {
  return conversion == 'd' || conversion == 'u' || conversion == 'x' || conversion == 'f';
}

std::vector<char> ParsePrintfConversions(const std::string& format) {
  std::vector<char> conversions;
  size_t i = 0;
  while (i < format.size()) {
    if (format[i] != '%') {
      ++i;
      continue;
    }
    if (i + 1 < format.size() && format[i + 1] == '%') {
      i += 2;
      continue;
    }

    size_t j = i + 1;
    while (j < format.size()) {
      char c = format[j];
      if (c == '-' || c == '+' || c == ' ' || c == '#' || c == '0') {
        ++j;
      } else {
        break;
      }
    }
    while (j < format.size() && std::isdigit(static_cast<unsigned char>(format[j]))) {
      ++j;
    }
    if (j < format.size() && format[j] == '.') {
      ++j;
      CHECK(j < format.size() && std::isdigit(static_cast<unsigned char>(format[j])))
          << "debug.printf precision must be followed by digits";
      while (j < format.size() && std::isdigit(static_cast<unsigned char>(format[j]))) {
        ++j;
      }
    }

    CHECK(j < format.size()) << "debug.printf format ends with an incomplete conversion";
    char conversion = format[j];
    CHECK(IsSupportedPrintfConversion(conversion))
        << "debug.printf does not support conversion '%" << conversion << "'";
    conversions.push_back(conversion);
    i = j + 1;
  }

  CHECK(!conversions.empty()) << "debug.printf format must contain at least one supported conversion";
  return conversions;
}

bool IsPrintfIntegerType(const DataType& dtype) {
  return dtype == DataType::INDEX || dtype.IsInt();
}

}  // namespace

TypePtr DeduceDebugPrintfType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(!args.empty()) << "debug.printf requires at least one scalar argument";

  bool found_format = false;
  std::string format;
  for (const auto& [key, value] : kwargs) {
    if (key == "format") {
      format = AnyCast<std::string>(value, "kwarg key: format");
      found_format = true;
      break;
    }
  }
  CHECK(found_format) << "debug.printf requires 'format' kwarg";

  auto conversions = ParsePrintfConversions(format);
  CHECK(conversions.size() == args.size()) << "debug.printf format expects " << conversions.size()
                                           << " scalar arguments, but got " << args.size();

  for (size_t i = 0; i < args.size(); ++i) {
    auto scalar_type = As<ScalarType>(args[i]->GetType());
    CHECK(scalar_type) << "debug.printf argument " << i << " must be ScalarType, but got "
                       << args[i]->GetType()->TypeName();

    const DataType& dtype = scalar_type->dtype_;
    char conversion = conversions[i];
    if (conversion == 'f') {
      CHECK(dtype == DataType::FP32)
          << "debug.printf conversion '%f' requires FP32 scalar, but got " << dtype.ToString();
    } else {
      CHECK(IsPrintfIntegerType(dtype))
          << "debug.printf conversion '%" << conversion << "' requires integer/index scalar, but got "
          << dtype.ToString();
    }
  }

  return GetUnknownType();
}

REGISTER_OP("debug.printf")
    .set_op_category("DebugOp")
    .set_description("Print scalar values using a compile-time format string")
    .set_attr<std::string>("format")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceDebugPrintfType(args, kwargs);
    });

}  // namespace ir
}  // namespace pypto
