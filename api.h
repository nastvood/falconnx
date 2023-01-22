#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <onnxruntime_c_api.h>

#define ORT_RETURN_ON_ERROR(expr)                                   \
    do                                                              \
    {                                                               \
        OrtStatus *status = (expr);                                 \
        if (status != NULL)                                         \
        {                                                           \
            const char *msgStatus = g_ort->GetErrorMessage(status); \
            char *msgErr = calloc(strlen(msgStatus), sizeof(char)); \
            strcpy(msgErr, msgStatus);                              \
            g_ort->ReleaseStatus(status);                           \
            return msgErr;                                          \
        }                                                           \
    } while (0);

char *getOnnxTypeFromTypeInfo(const OrtApi *g_ort, OrtTypeInfo *typeInfo, enum ONNXType *out);
char *getTensorElementType(const OrtApi *g_ort, const OrtTensorTypeAndShapeInfo *info, enum ONNXTensorElementDataType *out);
char *getTensorShapeElementCount(const OrtApi *g_ort, const OrtTensorTypeAndShapeInfo *info, size_t *out);
char *getDimensionsCount(const OrtApi *g_ort, const OrtTensorTypeAndShapeInfo *info, size_t *out);
char *getDimensions(const OrtApi *g_ort, const OrtTensorTypeAndShapeInfo *info, int64_t *dim_values, size_t dim_values_length);

char *castTypeInfoToTensorInfo(const OrtApi *g_ort, OrtTypeInfo *typeInfo, const OrtTensorTypeAndShapeInfo **out);
char *castTypeInfoToMapTypeInfo(const OrtApi *g_ort, OrtTypeInfo *typeInfo, const OrtMapTypeInfo **out);
char *castTypeInfoToSequenceTypeInfo(const OrtApi *g_ort, OrtTypeInfo *typeInfo, const OrtSequenceTypeInfo **out);

void releaseTypeInfo(const OrtApi *g_ort, OrtTypeInfo *type_info);
void releaseTensorTypeInfo(const OrtApi *g_ort, OrtTensorTypeAndShapeInfo *info);