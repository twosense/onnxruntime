// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/rewrite_rule.h"
using namespace ::onnxruntime::common;

namespace onnxruntime {
// Manages a list of graph transformers. It is initialized with a list of graph
// transformers. Each inference session can further register additional ones.
class GraphTransformerManager {
 public:
  explicit GraphTransformerManager(unsigned steps) : steps_(steps) {
    
  }  

  // Register a transformer with a level and compatible providers list
  // If a transformer is execution provider independent add empty string in providers {""}
  common::Status Register(std::unique_ptr<GraphTransformer> transformer, const TransformerLevel& level, std::vector<std::string>&& provider);  

  // Apply all transformers registered for the given level on the given graph
  // Only transformers which are compatible with the given providers list will be applied
  // When applying transformers before partitioning (.i.e no execution providers are assigned yet) add empty string in providers {""}
  common::Status ApplyTransformers(Graph& graph, std::vector<std::string> providers, const TransformerLevel& level) const;

 private:
  GraphTransformerManager() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphTransformerManager);

  std::vector<std::string> GenerateCompatibleProvidersList(const std::vector<std::string>& provider_types, const std::string& name) const {
    std::vector<std::string> compatible_providers;
    const auto& entry = transformers_info_.find(name);
    ORT_ENFORCE(entry != transformers_info_.end());

    std::set_intersection(provider_types.begin(), provider_types.end(),
                          entry->second.compatible_providers.begin(), entry->second.compatible_providers.end(),
                          std::back_inserter(compatible_providers));
    
    return compatible_providers;
  }    

  const unsigned steps_;

  struct TransformerInfo {
   public:
    TransformerInfo(TransformerLevel level, std::vector<std::string>&& providers, GraphTransformer* graphTransformer)
        : level{level}, compatible_providers{std::move(providers)}, transformer{graphTransformer} {}
    TransformerInfo() = default;

    TransformerLevel level;
    std::vector<std::string> compatible_providers;
    GraphTransformer* transformer;
  };

  std::unordered_map<TransformerLevel, std::vector<std::unique_ptr<GraphTransformer>>> level_to_transformer_map_;
  std::unordered_map<std::string, TransformerInfo> transformers_info_;  

};
}  // namespace onnxruntime