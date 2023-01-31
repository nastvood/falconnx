#include "api.h"

// --------------------- GET ------------------------------

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

char *createEnv(const OrtApi *g_ort, OrtLoggingLevel level, const char *logid, OrtEnv **env)
{
    ORT_RETURN_ON_ERROR(g_ort->CreateEnv(level, logid, env));

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

char *createTensorWithDataAsOrtValue(const OrtApi *g_ort, OrtMemoryInfo *memory_info, void *input, size_t input_len, int64_t *shape, size_t shape_len, ONNXTensorElementDataType type, OrtValue **out)
{
    size_t model_input_len = input_len;

    switch (type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        model_input_len *= sizeof(float);
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        model_input_len *= sizeof(uint8_t);
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        model_input_len *= sizeof(int8_t);
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        model_input_len *= sizeof(uint16_t);
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        model_input_len *= sizeof(int16_t);
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        model_input_len *= sizeof(int32_t);
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        model_input_len *= sizeof(int64_t);
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        RETURN_ERROR_MSG("not implemented for ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING")
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        RETURN_ERROR_MSG("not implemented for ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL")
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        RETURN_ERROR_MSG("not implemented for ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16")
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        model_input_len *= sizeof(double);
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        model_input_len *= sizeof(uint32_t);
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        model_input_len *= sizeof(uint64_t);
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
        RETURN_ERROR_MSG("not implemented for ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64")
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
        RETURN_ERROR_MSG("not implemented for ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128")
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        RETURN_ERROR_MSG("not implemented for ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16")
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
        RETURN_ERROR_MSG("not implemented for ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED")
        break;
    }

    ORT_RETURN_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input, model_input_len, shape, shape_len, type, out));

    return NULL;
}

// --------------------- GET ------------------------------

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

char *getValueType(const OrtApi *g_ort, const OrtValue *value, enum ONNXType *out)
{
    ORT_RETURN_ON_ERROR(g_ort->GetValueType(value, out));

    return NULL;
}

char *getAvailableProviders(const OrtApi *g_ort)
{
    char **output = (char **)calloc(1, sizeof(char **));
    int len = 0;
    ORT_RETURN_ON_ERROR(g_ort->GetAvailableProviders(&output, &len));

    printf("%d\n", len);
    for (int i = 0; i < len; ++i)
    {
        printf("%s\n", output[i]);
    }

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

void releaseAllocatorArrayOfString(OrtAllocator *allocator, size_t size, char **strings)
{
    for (size_t i = 0; i < size; i++)
    {
        allocator->Free(allocator, strings[i]);
    }

    allocator->Free(allocator, (strings));
}

// --------------------- RUN ------------------------------

char *run(const OrtApi *g_ort, OrtSession *session, OrtMemoryInfo *memory_info, OrtAllocator *allocator,
          char **input_names, size_t input_names_len, OrtValue *input_value,
          char **output_names, size_t output_names_len, OrtValue **outputs)
{
    ORT_RETURN_ON_ERROR(g_ort->Run(session, NULL, (const char *const *)input_names, (const OrtValue *const *)&input_value, input_names_len, (const char *const *)output_names, output_names_len, outputs));

    return NULL;
}