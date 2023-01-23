#include "api.h"

char *getOnnxTypeFromTypeInfo(const OrtApi *g_ort, OrtTypeInfo *typeInfo, enum ONNXType *out)
{
    ORT_RETURN_ON_ERROR(g_ort->GetOnnxTypeFromTypeInfo(typeInfo, out));

    return NULL;
}

char *castTypeInfoToTensorInfo(const OrtApi *g_ort, OrtTypeInfo *typeInfo, const OrtTensorTypeAndShapeInfo **out)
{
    ORT_RETURN_ON_ERROR(g_ort->CastTypeInfoToTensorInfo(typeInfo, out));

    return NULL;
}

char *castTypeInfoToMapTypeInfo(const OrtApi *g_ort, OrtTypeInfo *typeInfo, const OrtMapTypeInfo **out)
{
    ORT_RETURN_ON_ERROR(g_ort->CastTypeInfoToMapTypeInfo(typeInfo, out));

    return NULL;
}

char *castTypeInfoToSequenceTypeInfo(const OrtApi *g_ort, OrtTypeInfo *typeInfo, const OrtSequenceTypeInfo **out)
{
    ORT_RETURN_ON_ERROR(g_ort->CastTypeInfoToSequenceTypeInfo(typeInfo, out));

    return NULL;
}

char *getTensorElementType(const OrtApi *g_ort, const OrtTensorTypeAndShapeInfo *info, enum ONNXTensorElementDataType *out)
{
    ORT_RETURN_ON_ERROR(g_ort->GetTensorElementType(info, out));

    return NULL;
}

char *getTensorShapeElementCount(const OrtApi *g_ort, const OrtTensorTypeAndShapeInfo *info, size_t *out)
{
    ORT_RETURN_ON_ERROR(g_ort->GetTensorShapeElementCount(info, out));

    return NULL;
}

char *getDimensionsCount(const OrtApi *g_ort, const OrtTensorTypeAndShapeInfo *info, size_t *out)
{
    ORT_RETURN_ON_ERROR(g_ort->GetDimensionsCount(info, out));

    return NULL;
}

char *getDimensions(const OrtApi *g_ort, const OrtTensorTypeAndShapeInfo *info, int64_t *dim_values, size_t dim_values_length)
{
    ORT_RETURN_ON_ERROR(g_ort->GetDimensions(info, dim_values, dim_values_length))

    return NULL;
}

char *getTypeInfo(const OrtApi *g_ort, const OrtValue *value, OrtTypeInfo **out)
{
    ORT_RETURN_ON_ERROR(g_ort->GetTypeInfo(value, out));

    return NULL;
}

char *getSequenceElementType(const OrtApi *g_ort, const OrtSequenceTypeInfo *sequence_type_info, OrtTypeInfo **out)
{
    ORT_RETURN_ON_ERROR(g_ort->GetSequenceElementType(sequence_type_info, out));

    return NULL;
}

char *getMapKeyType(const OrtApi *g_ort, const OrtMapTypeInfo *map_type_info, enum ONNXTensorElementDataType *out)
{
    ORT_RETURN_ON_ERROR(g_ort->GetMapKeyType(map_type_info, out));

    return NULL;
}

char *getMapValueType(const OrtApi *g_ort, const OrtMapTypeInfo *map_type_info, OrtTypeInfo **out)
{
    ORT_RETURN_ON_ERROR(g_ort->GetMapValueType(map_type_info, out));

    return NULL;
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

void releaseSequenceTypeInfo(const OrtApi *g_ort, OrtSequenceTypeInfo *info)
{
    g_ort->ReleaseSequenceTypeInfo(info);
}

void releaseMapTypeInfo(const OrtApi *g_ort, OrtMapTypeInfo *info)
{
    g_ort->ReleaseMapTypeInfo(info);
}