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

const OrtApi *createApi();
char *createMemoryInfo(const OrtApi *g_ort, OrtMemoryInfo **memory_info);
char *createEnv(const OrtApi *g_ort, OrtLoggingLevel level, const char *logid, OrtEnv **env);
char *createSession(const OrtApi *g_ort, OrtEnv *env, OrtSessionOptions **session_options, OrtSession **session, const char *model_path);
char *createAllocator(const OrtApi *g_ort, OrtSession *session, OrtMemoryInfo *memory_info, OrtAllocator **allocator);
char *createTensorWithDataAsOrtValue(const OrtApi *g_ort, OrtMemoryInfo *memory_info, void *input, size_t input_len, int64_t *shape, size_t shape_len, ONNXTensorElementDataType type, OrtValue **out);

char *getInputCount(const OrtApi *g_ort, OrtSession *session, size_t *input_count);
char *getInputNames(const OrtApi *g_ort, OrtSession *session, OrtAllocator *allocator, size_t input_count, char ***out);
char *getInputInfo(const OrtApi *g_ort, OrtSession *session, size_t index, OrtTypeInfo **type_info);
char *getOutputCount(const OrtApi *g_ort, OrtSession *session, size_t *output_count);
char *getOutputNames(const OrtApi *g_ort, OrtSession *session, OrtAllocator *allocator, size_t output_count, char ***out);
char *getOutputInfo(const OrtApi *g_ort, OrtSession *session, size_t index, OrtTypeInfo **type_info);

char *getOnnxTypeFromTypeInfo(const OrtApi *g_ort, OrtTypeInfo *typeInfo, enum ONNXType *out);
char *getTypeInfo(const OrtApi *g_ort, const OrtValue *value, OrtTypeInfo **out);

char *getTensorElementType(const OrtApi *g_ort, const OrtTensorTypeAndShapeInfo *info, enum ONNXTensorElementDataType *out);
char *getTensorShapeElementCount(const OrtApi *g_ort, const OrtTensorTypeAndShapeInfo *info, size_t *out);
char *getDimensionsCount(const OrtApi *g_ort, const OrtTensorTypeAndShapeInfo *info, size_t *out);
char *getDimensions(const OrtApi *g_ort, const OrtTensorTypeAndShapeInfo *info, int64_t *dim_values, size_t dim_values_length);

char *getSequenceElementType(const OrtApi *g_ort, const OrtSequenceTypeInfo *sequence_type_info, OrtTypeInfo **out);

char *getMapKeyType(const OrtApi *g_ort, const OrtMapTypeInfo *map_type_info, enum ONNXTensorElementDataType *out);
char *getMapValueType(const OrtApi *g_ort, const OrtMapTypeInfo *map_type_info, OrtTypeInfo **out);

char *castTypeInfoToTensorInfo(const OrtApi *g_ort, OrtTypeInfo *typeInfo, const OrtTensorTypeAndShapeInfo **out);
char *castTypeInfoToSequenceTypeInfo(const OrtApi *g_ort, OrtTypeInfo *typeInfo, const OrtSequenceTypeInfo **out);
char *castTypeInfoToMapTypeInfo(const OrtApi *g_ort, OrtTypeInfo *typeInfo, const OrtMapTypeInfo **out);

char *getTensorMutableData(const OrtApi *g_ort, OrtValue *value, void **out);
char *getValue(const OrtApi *g_ort, OrtAllocator *allocator, OrtValue *value, int index, OrtValue **out);
char *getValueCount(const OrtApi *g_ort, OrtValue *value, size_t *out);

void releaseEnv(const OrtApi *g_ort, OrtEnv *env);
void releaseMapTypeInfo(const OrtApi *g_ort, OrtMapTypeInfo *info);
void releaseSession(const OrtApi *g_ort, OrtSession *session);
void releaseSessionOptions(const OrtApi *g_ort, OrtSessionOptions *sessionOptions);
void releaseAllocator(const OrtApi *g_ort, OrtAllocator *allocator);
void releaseMemoryInfo(const OrtApi *g_ort, OrtMemoryInfo *memory_info);
void releaseTypeInfo(const OrtApi *g_ort, OrtTypeInfo *type_info);
void releaseTensorTypeInfo(const OrtApi *g_ort, OrtTensorTypeAndShapeInfo *info);
void releaseSequenceTypeInfo(const OrtApi *g_ort, OrtSequenceTypeInfo *info);
void releaseValue(const OrtApi *g_ort, OrtValue *value);
void releaseAllocatorArrayOfString(OrtAllocator *allocator, size_t size, char **strings);

char *run(const OrtApi *g_ort, OrtSession *session, OrtMemoryInfo *memory_info, OrtAllocator *allocator,
          char **input_names, size_t input_names_len, OrtValue *input_value,
          char **output_names, size_t output_names_len, OrtValue **outputs);