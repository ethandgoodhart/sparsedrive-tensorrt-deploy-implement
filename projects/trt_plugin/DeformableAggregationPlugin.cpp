#include "DeformableAggregationPlugin.h"
#include <cuda_runtime.h>

extern "C" void DeformableAggregationLauncher(
    const void* mc_ms_feat, const int* spatial_shape, const int* scale_start_index,
    const void* sample_location, const void* weights, void* output,
    int batch_size, int num_cams, int num_feat, int num_embeds,
    int num_scale, int num_anchors, int num_pts, int num_groups,
    bool is_cam_shared,
    bool is_fp16,
    cudaStream_t stream
);

extern "C" void CastInt64ToInt32(const void* in, void* out, int n, cudaStream_t stream);

namespace {
int64_t volume(nvinfer1::Dims const& d) {
    int64_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= d.d[i];
    return v;
}
}

namespace nvinfer1 {

REGISTER_TENSORRT_PLUGIN(DeformableAggregationPluginCreator);

// ---------------- IPluginV3 ----------------

IPluginCapability* DeformableAggregationPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept {
    switch (type) {
        case PluginCapabilityType::kCORE:    return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD:   return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
    }
    return nullptr;
}

IPluginV3* DeformableAggregationPlugin::clone() noexcept {
    auto* p = new (std::nothrow) DeformableAggregationPlugin();
    if (p) p->setPluginNamespace(mNamespace.c_str());
    return p;
}

// ---------------- IPluginV3OneBuild ----------------

int32_t DeformableAggregationPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs,
    DataType const* inputTypes, int32_t nbInputs) const noexcept {
    if (nbOutputs < 1 || nbInputs < 1 || outputTypes == nullptr || inputTypes == nullptr) return -1;
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t DeformableAggregationPlugin::getOutputShapes(
    DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /*shapeInputs*/, int32_t /*nbShapeInputs*/,
    DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& /*exprBuilder*/) noexcept {
    if (nbInputs < 4 || nbOutputs < 1 || inputs == nullptr || outputs == nullptr) return -1;
    DimsExprs& out = outputs[0];
    out.nbDims = 3;
    out.d[0] = inputs[0].d[0];                                   // Batch
    out.d[1] = inputs[3].d[1];                                   // Num Anchors
    out.d[2] = inputs[0].d[inputs[0].nbDims - 1];                // Embed dim
    return 0;
}

bool DeformableAggregationPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut,
    int32_t /*nbInputs*/, int32_t /*nbOutputs*/) noexcept {
    auto const& desc = inOut[pos].desc;
    if (desc.format != TensorFormat::kLINEAR) return false;
    // Inputs 1 and 2 carry shape / scale-start-index. TRT 10's ONNX parser emits
    // INT64 by default, but TRT 8 used INT32 — accept either.
    if (pos == 1 || pos == 2) {
        return desc.type == DataType::kINT32 || desc.type == DataType::kINT64;
    }
    // input 0/3/4 and output 0 are float-like. Force FP16 — kernel is memory-bound
    // and FP32 leaves performance on the table. TRT will insert Reformat nodes if
    // upstream/downstream is FP32; the savings inside the plugin more than pay.
    return desc.type == DataType::kHALF;
}

size_t DeformableAggregationPlugin::getWorkspaceSize(
    DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept {
    size_t ws = 0;
    if (nbInputs > 2) {
        if (inputs[1].desc.type == DataType::kINT64) ws += volume(inputs[1].desc.dims) * sizeof(int);
        if (inputs[2].desc.type == DataType::kINT64) ws += volume(inputs[2].desc.dims) * sizeof(int);
    }
    // 256-byte align in case TRT layers two allocations
    return (ws + 255) & ~static_cast<size_t>(255);
}

// ---------------- IPluginV3OneRuntime ----------------

int32_t DeformableAggregationPlugin::enqueue(
    PluginTensorDesc const* inputDesc, PluginTensorDesc const* /*outputDesc*/,
    void const* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept {
    int dims0 = inputDesc[0].dims.nbDims;
    int num_embeds = inputDesc[0].dims.d[dims0 - 1];
    int num_feat   = inputDesc[0].dims.d[dims0 - 2];
    int batch_size = (dims0 >= 3) ? inputDesc[0].dims.d[dims0 - 3] : 1;

    int dims4 = inputDesc[4].dims.nbDims;
    int num_groups  = inputDesc[4].dims.d[dims4 - 1];
    int num_scale   = inputDesc[4].dims.d[dims4 - 2];
    int num_cams    = inputDesc[4].dims.d[dims4 - 3];
    int num_pts     = inputDesc[4].dims.d[dims4 - 4];
    int num_anchors = inputDesc[4].dims.d[dims4 - 5];

    int start_index_vol = 1;
    for (int i = 0; i < inputDesc[2].dims.nbDims; ++i) start_index_vol *= inputDesc[2].dims.d[i];
    bool is_cam_shared = (start_index_vol == num_scale);
    bool is_fp16 = (inputDesc[0].type == DataType::kHALF);

    // If TRT handed us INT64 shape/index tensors, cast into workspace.
    const int* spatial_shape_i32 = static_cast<const int*>(inputs[1]);
    const int* scale_start_i32   = static_cast<const int*>(inputs[2]);
    char* ws = static_cast<char*>(workspace);

    if (inputDesc[1].type == DataType::kINT64) {
        int n = 1;
        for (int i = 0; i < inputDesc[1].dims.nbDims; ++i) n *= inputDesc[1].dims.d[i];
        CastInt64ToInt32(inputs[1], ws, n, stream);
        spatial_shape_i32 = reinterpret_cast<int*>(ws);
        ws += static_cast<size_t>(n) * sizeof(int);
    }
    if (inputDesc[2].type == DataType::kINT64) {
        int n = 1;
        for (int i = 0; i < inputDesc[2].dims.nbDims; ++i) n *= inputDesc[2].dims.d[i];
        CastInt64ToInt32(inputs[2], ws, n, stream);
        scale_start_i32 = reinterpret_cast<int*>(ws);
    }

    DeformableAggregationLauncher(
        inputs[0], spatial_shape_i32, scale_start_i32,
        inputs[3], inputs[4], outputs[0],
        batch_size, num_cams, num_feat, num_embeds,
        num_scale, num_anchors, num_pts, num_groups,
        is_cam_shared, is_fp16, stream
    );
    return 0;
}

// ---------------- Creator ----------------

DeformableAggregationPluginCreator::DeformableAggregationPluginCreator() {
    mFC.nbFields = 0;
    mFC.fields = nullptr;
    mNamespace = "";
}

IPluginV3* DeformableAggregationPluginCreator::createPlugin(
    AsciiChar const* /*name*/, PluginFieldCollection const* /*fc*/,
    TensorRTPhase /*phase*/) noexcept {
    auto* p = new (std::nothrow) DeformableAggregationPlugin();
    if (p) p->setPluginNamespace(mNamespace.c_str());
    return p;
}

} // namespace nvinfer1
