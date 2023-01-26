#include "api.h"

const OrtApi *createApi()
{
    const OrtApi *g_ort = NULL;
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    return g_ort;
}

char *createMemoryInfo(const OrtApi *g_ort, OrtMemoryInfo **memory_info)
{
    ORT_RETURN_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, memory_info));

    return NULL;
}

char *createEnv(const OrtApi *g_ort, OrtEnv **env)
{
    ORT_RETURN_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", env));

    return NULL;
}

char *createSession(const OrtApi *g_ort, OrtEnv *env, OrtSessionOptions **session_options, OrtSession **session, const char *model_path)
{
    ORT_RETURN_ON_ERROR(g_ort->CreateSessionOptions(session_options));
    ORT_RETURN_ON_ERROR(g_ort->CreateSession(env, model_path, *session_options, session));

    return NULL;
}

char *createAllocator(const OrtApi *g_ort, OrtSession *session, OrtMemoryInfo *memory_info, OrtAllocator **allocator)
{
    ORT_RETURN_ON_ERROR(g_ort->CreateAllocator(session, memory_info, allocator));

    return NULL;
}

// --------------------- CREATE ------------------------------

char *getInputCount(const OrtApi *g_ort, OrtSession *session, size_t *input_count)
{
    ORT_RETURN_ON_ERROR(g_ort->SessionGetInputCount(session, input_count));

    return NULL;
}

char *getInputNames(const OrtApi *g_ort, OrtSession *session, OrtAllocator *allocator, size_t input_count, char ***out)
{
    char **input_names = allocator->Alloc(allocator, input_count * sizeof(char *));
    for (size_t i = 0; i < input_count; ++i)
    {
        ORT_RETURN_ON_ERROR(g_ort->SessionGetInputName(session, i, allocator, &input_names[i]));
    }

    *out = input_names;

    return NULL;
}

char *getInputInfo(const OrtApi *g_ort, OrtSession *session, size_t index, OrtTypeInfo **type_info)
{
    ORT_RETURN_ON_ERROR(g_ort->SessionGetInputTypeInfo(session, index, type_info));

    return NULL;
}

char *getOutputCount(const OrtApi *g_ort, OrtSession *session, size_t *output_count)
{
    ORT_RETURN_ON_ERROR(g_ort->SessionGetOutputCount(session, output_count));

    return NULL;
}

char *getOutputNames(const OrtApi *g_ort, OrtSession *session, OrtAllocator *allocator, size_t output_count, char ***out)
{
    char **output_names = allocator->Alloc(allocator, output_count * sizeof(char *));
    for (size_t i = 0; i < output_count; ++i)
    {
        ORT_RETURN_ON_ERROR(g_ort->SessionGetOutputName(session, i, allocator, &output_names[i]));
    }

    *out = output_names;

    return NULL;
}

char *getOutputInfo(const OrtApi *g_ort, OrtSession *session, size_t index, OrtTypeInfo **type_info)
{
    ORT_RETURN_ON_ERROR(g_ort->SessionGetOutputTypeInfo(session, index, type_info));

    return NULL;
}

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

char *getTensorMutableData(const OrtApi *g_ort, OrtValue *value, void **out)
{

    ORT_RETURN_ON_ERROR(g_ort->GetTensorMutableData(value, out));

    return NULL;
}

char *getValue(const OrtApi *g_ort, OrtAllocator *allocator, OrtValue *value, int index, OrtValue **out)
{
    ORT_RETURN_ON_ERROR(g_ort->GetValue(value, index, allocator, out));

    return NULL;
}

char *getValueCount(const OrtApi *g_ort, OrtValue *value, size_t *out)
{

    ORT_RETURN_ON_ERROR(g_ort->GetValueCount(value, out));

    return NULL;
}

// --------------------- RELEASES ------------------------------

void releaseEnv(const OrtApi *g_ort, OrtEnv *env)
{
    g_ort->ReleaseEnv(env);
}

void releaseMemoryInfo(const OrtApi *g_ort, OrtMemoryInfo *memory_info)
{
    g_ort->ReleaseMemoryInfo(memory_info);
}

void releaseSession(const OrtApi *g_ort, OrtSession *session)
{
    g_ort->ReleaseSession(session);
}

void releaseSessionOptions(const OrtApi *g_ort, OrtSessionOptions *sessionOptions)
{
    g_ort->ReleaseSessionOptions(sessionOptions);
}

void releaseAllocator(const OrtApi *g_ort, OrtAllocator *allocator)
{
    g_ort->ReleaseAllocator(allocator);
}

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

void releaseValue(const OrtApi *g_ort, OrtValue *value)
{
    g_ort->ReleaseValue(value);
}