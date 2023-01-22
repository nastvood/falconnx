#include "api.h"

char *getOnnxTypeFromTypeInfo(const OrtApi *g_ort, OrtTypeInfo *typeInfo, enum ONNXType *out)
{
    ORT_RETURN_ON_ERROR(g_ort->GetOnnxTypeFromTypeInfo(typeInfo, out));
}

char *castTypeInfoToTensorInfo(const OrtApi *g_ort, OrtTypeInfo *typeInfo, const OrtTensorTypeAndShapeInfo **out)
{
    ORT_RETURN_ON_ERROR(g_ort->CastTypeInfoToTensorInfo(typeInfo, out));
}

char *castTypeInfoToMapTypeInfo(const OrtApi *g_ort, OrtTypeInfo *typeInfo, const OrtMapTypeInfo **out)
{
    ORT_RETURN_ON_ERROR(g_ort->CastTypeInfoToMapTypeInfo(typeInfo, out));
}

char *castTypeInfoToSequenceTypeInfo(const OrtApi *g_ort, OrtTypeInfo *typeInfo, const OrtSequenceTypeInfo **out)
{
    ORT_RETURN_ON_ERROR(g_ort->CastTypeInfoToSequenceTypeInfo(typeInfo, out));
}

char *getTensorElementType(const OrtApi *g_ort, const OrtTensorTypeAndShapeInfo *info, enum ONNXTensorElementDataType *out)
{
    ORT_RETURN_ON_ERROR(g_ort->GetTensorElementType(info, out));
}

char *getTensorShapeElementCount(const OrtApi *g_ort, const OrtTensorTypeAndShapeInfo *info, size_t *out)
{
    ORT_RETURN_ON_ERROR(g_ort->GetTensorShapeElementCount(info, out));
}

char *getDimensionsCount(const OrtApi *g_ort, const OrtTensorTypeAndShapeInfo *info, size_t *out)
{
    ORT_RETURN_ON_ERROR(g_ort->GetDimensionsCount(info, out));
}

char *getDimensions(const OrtApi *g_ort, const OrtTensorTypeAndShapeInfo *info, int64_t *dim_values, size_t dim_values_length)
{
    // int64_t dim_values[] = {0, 0};
    ORT_RETURN_ON_ERROR(g_ort->GetDimensions(info, dim_values, dim_values_length))
}
// --------------------- RELEASES ------------------------------

void releaseTypeInfo(const OrtApi *g_ort, OrtTypeInfo *type_info)
{
    g_ort->ReleaseTypeInfo(type_info);
}

void releaseTensorTypeInfo(const OrtApi *g_ort, OrtTensorTypeAndShapeInfo *info)
{
    g_ort->ReleaseTensorTypeAndShapeInfo(info);
}