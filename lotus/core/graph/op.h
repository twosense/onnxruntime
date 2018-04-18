﻿#pragma once

#include <functional>
#include <unordered_map>
#include "onnx/defs/schema.h"
#include "core/common/status.h"
#include "core/graph/constants.h"

using namespace onnx;
using namespace Lotus::Common;

namespace LotusIR {
typedef AttributeProto_AttributeType AttrType;
typedef std::unordered_map<std::string, AttributeProto> NodeAttributes;

// This string array should exactly match the AttrType defined above.
/*
AttributeProto_AttributeType_UNDEFINED = 0,
AttributeProto_AttributeType_FLOAT = 1,
AttributeProto_AttributeType_INT = 2,
AttributeProto_AttributeType_STRING = 3,
AttributeProto_AttributeType_TENSOR = 4,
AttributeProto_AttributeType_GRAPH = 5,
AttributeProto_AttributeType_FLOATS = 6,
AttributeProto_AttributeType_INTS = 7,
AttributeProto_AttributeType_STRINGS = 8,
AttributeProto_AttributeType_TENSORS = 9,
AttributeProto_AttributeType_GRAPHS = 10
*/
static constexpr const char* kAttrTypeStrings[] =
    {
        "UNDEFINED",
        "FLOAT",
        "INT",
        "STRING",
        "TENSOR",
        "GRAPH",
        "FLOATS",
        "INTS",
        "STRINGS",
        "TENSORS",
        "GRAPHS"};

class TypeUtils {
 public:
  // Get attribute type given attribute proto data.
  static Status GetType(const AttributeProto& attr, AttrType& type);
  static bool IsValidAttribute(const AttributeProto& attribute);
};

class MsOpRegistry {
 public:
  static Status RegisterMsOps() {
    RETURN_IF_ERROR(RegisterMsNNOps());
    return Status::OK();
  }

 private:
  static Status RegisterMsNNOps();
};

}  // namespace LotusIR
