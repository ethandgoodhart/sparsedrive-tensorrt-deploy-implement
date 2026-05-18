#pragma once
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "NvInferRuntimePlugin.h"
#include "NvInferPluginBase.h"
#include <string>

namespace nvinfer1 {

class DeformableAggregationPlugin : public IPluginV3,
                                    public IPluginV3OneCore,
                                    public IPluginV3OneBuild,
                                    public IPluginV3OneRuntime {
public:
    DeformableAggregationPlugin() = default;
    ~DeformableAggregationPlugin() override = default;

    // IPluginV3
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override;
    IPluginV3* clone() noexcept override;

    // IPluginV3OneCore
    AsciiChar const* getPluginName() const noexcept override { return "DeformableAggregation"; }
    AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
    AsciiChar const* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }
    void setPluginNamespace(AsciiChar const* ns) noexcept { mNamespace = ns ? ns : ""; }

    // IPluginV3OneBuild
    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t configurePlugin(DynamicPluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
                            DynamicPluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept override {
        return 0;
    }

    int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
                               DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
                            DimsExprs const* shapeInputs, int32_t nbShapeInputs,
                            DimsExprs* outputs, int32_t nbOutputs,
                            IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut,
                                   int32_t nbInputs, int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
                            DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    // IPluginV3OneRuntime
    int32_t onShapeChange(PluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
                          PluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept override {
        return 0;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                    void const* const* inputs, void* const* outputs,
                    void* workspace, cudaStream_t stream) noexcept override;

    IPluginV3* attachToContext(IPluginResourceContext* /*context*/) noexcept override {
        return clone();
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        mFCToSerialize.nbFields = 0;
        mFCToSerialize.fields = nullptr;
        return &mFCToSerialize;
    }

private:
    std::string mNamespace;
    PluginFieldCollection mFCToSerialize{};
};

class DeformableAggregationPluginCreator : public IPluginCreatorV3One {
public:
    DeformableAggregationPluginCreator();
    ~DeformableAggregationPluginCreator() override = default;

    AsciiChar const* getPluginName() const noexcept override { return "DeformableAggregation"; }
    AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &mFC; }

    IPluginV3* createPlugin(AsciiChar const* name, PluginFieldCollection const* fc,
                            TensorRTPhase phase) noexcept override;

    AsciiChar const* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }
    void setPluginNamespace(AsciiChar const* ns) noexcept { mNamespace = ns ? ns : ""; }

private:
    PluginFieldCollection mFC{};
    std::string mNamespace;
};

} // namespace nvinfer1
