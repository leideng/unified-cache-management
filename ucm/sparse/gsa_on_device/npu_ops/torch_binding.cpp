/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>
#include <torch/library.h>
#include <torch/version.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include <torch_npu/csrc/npu/Module.h>
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "ops.h"
#include "utils.h"
#include "mla_preprocess/op_host/mla_preprocess.h"
#include "batch_matmul_transpose/op_host/batch_matmul_transpose.h"
#include "aclnn_torch_adapter/op_api_common.h"

#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>

namespace vllm_ascend {

at::Tensor convert_hamming_dist_top_k_output(const at::Tensor &hashq,
                                             const at::Tensor &hashkCache,
                                             const c10::optional<at::Tensor>& indices) {
    if (indices.has_value()) {
        return indices.value();
    }

    auto n_bs = hashq.size(0);
    auto n_kv_heads = hashkCache.size(1);
    auto n_max_kv = 512; // 设置和hamming_dist_top_k算子实现一致
    at::Tensor res = at::empty({n_bs, n_kv_heads, n_max_kv}, torch::TensorOptions().dtype(torch::kInt32).device(hashq.device()));
    return res;
}

at::Tensor npu_hamming_dist_top_k(const at::Tensor &hashq,
                                 const at::Tensor &hashkCache,
                                 const at::Tensor &topN,
                                 const at::Tensor &seqLen,
                                 const c10::optional<at::Tensor> &chunkSize,
                                 const c10::optional<int64_t> maxSeqLen,
                                 const c10::optional<int64_t> sink,
                                 const c10::optional<int64_t> recent,
                                 const c10::optional<int64_t> supportOffload,
                                 const c10::optional<at::Tensor> &blockTable,
                                 const c10::optional<at::Tensor>& indices) {

    auto&& maxSeqLen_ = maxSeqLen.value_or(0);
    auto&& sink_ = sink.value_or(0);
    auto&& recent_ = recent.value_or(0);
    auto&& supportOffload_ = supportOffload.value_or(0);

    at::Tensor out = convert_hamming_dist_top_k_output(hashq, hashkCache, indices);
    EXEC_NPU_CMD(aclnnHammingDistTopK, hashq, hashkCache, topN, seqLen, chunkSize, blockTable, maxSeqLen_, sink_, recent_, supportOffload_, out);
    return out;
}

at::Tensor npu_reshape_and_cache_bnsd(const at::Tensor& hashq,
                                      const at::Tensor& hashkCache,
                                      const at::Tensor& slotMapping,
                                      const at::Tensor& seqLen,
                                      const at::Tensor& hashkCacheOut) {
    EXEC_NPU_CMD(aclnnReshapeAndCacheBnsd, hashq, hashkCache, slotMapping, seqLen, hashkCacheOut);
    return hashkCacheOut;
}

} // namespace vllm_ascend

TORCH_LIBRARY_EXPAND(CONCAT(_C, _ascend), ops)
{
    ops.def(
        "npu_hamming_dist_top_k(Tensor q, Tensor k_comp, Tensor k,"
        "                      Tensor seq_len, Tensor? chunk_size=None,"
        "                      int? max_seq_len=None, int? sink=None, int? recent=None, int? support_offload=None,"
        "                      Tensor? key_block_table=None, Tensor? indices=None) -> Tensor"
    );
    ops.impl("npu_hamming_dist_top_k", torch::kPrivateUse1, &vllm_ascend::npu_hamming_dist_top_k);

    ops.def(
        "npu_reshape_and_cache_bnsd(Tensor q, Tensor k_comp, Tensor slot_mapping, Tensor seq_len, Tensor k_out) -> Tensor"
    );
    ops.impl("npu_reshape_and_cache_bnsd", torch::kPrivateUse1, &vllm_ascend::npu_reshape_and_cache_bnsd);
}
