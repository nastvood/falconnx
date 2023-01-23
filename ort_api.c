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

static const OrtApi *createApi()
{
    const OrtApi *g_ort = NULL;
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    return g_ort;
}

static char *createMemoryInfo(const OrtApi *g_ort, OrtMemoryInfo **memory_info)
{
    ORT_RETURN_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, memory_info));

    return NULL;
}

static char *createEnv(const OrtApi *g_ort, OrtEnv **env)
{
    ORT_RETURN_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", env));

    return NULL;
}

static void releaseEnv(const OrtApi *g_ort, OrtEnv *env)
{
    g_ort->ReleaseEnv(env);
}

static char *createSession(const OrtApi *g_ort, OrtEnv *env, OrtSessionOptions **session_options, OrtSession **session, const char *model_path)
{
    ORT_RETURN_ON_ERROR(g_ort->CreateSessionOptions(session_options));
    ORT_RETURN_ON_ERROR(g_ort->CreateSession(env, model_path, *session_options, session));

    return NULL;
}

static char *createAllocator(const OrtApi *g_ort, OrtSession *session, OrtMemoryInfo *memory_info, OrtAllocator **allocator)
{
    ORT_RETURN_ON_ERROR(g_ort->CreateAllocator(session, memory_info, allocator));

    return NULL;
}

static char *getInputCount(const OrtApi *g_ort, OrtSession *session, size_t *input_count)
{
    ORT_RETURN_ON_ERROR(g_ort->SessionGetInputCount(session, input_count));

    return NULL;
}

static char *getInputNames(const OrtApi *g_ort, OrtSession *session, OrtAllocator *allocator, size_t input_count, char ***out)
{
    char **input_names = allocator->Alloc(allocator, input_count * sizeof(char *));
    for (size_t i = 0; i < input_count; ++i)
    {
        ORT_RETURN_ON_ERROR(g_ort->SessionGetInputName(session, i, allocator, &input_names[i]));
    }

    *out = input_names;

    return NULL;
}

static char *getOutputCount(const OrtApi *g_ort, OrtSession *session, size_t *output_count)
{
    ORT_RETURN_ON_ERROR(g_ort->SessionGetOutputCount(session, output_count));

    return NULL;
}

static char *getOutputNames(const OrtApi *g_ort, OrtSession *session, OrtAllocator *allocator, size_t output_count, char ***out)
{
    char **output_names = allocator->Alloc(allocator, output_count * sizeof(char *));
    for (size_t i = 0; i < output_count; ++i)
    {
        ORT_RETURN_ON_ERROR(g_ort->SessionGetOutputName(session, i, allocator, &output_names[i]));
    }

    *out = output_names;

    return NULL;
}

static char *getInputInfo(const OrtApi *g_ort, OrtSession *session, size_t index, OrtTypeInfo **type_info)
{
    ORT_RETURN_ON_ERROR(g_ort->SessionGetInputTypeInfo(session, index, type_info));

    return NULL;
}

static void releaseSession(const OrtApi *g_ort, OrtSession *session)
{
    g_ort->ReleaseSession(session);
}

static void releaseSessionOptions(const OrtApi *g_ort, OrtSessionOptions *sessionOptions)
{
    g_ort->ReleaseSessionOptions(sessionOptions);
}

static void releaseAllocator(const OrtApi *g_ort, OrtAllocator *allocator)
{
    g_ort->ReleaseAllocator(allocator);
}

static void releaseValue(const OrtApi *g_ort, OrtValue *value)
{
    g_ort->ReleaseValue(value);
}

static void releaseAllocatorArrayOfString(OrtAllocator *allocator, size_t size, char **strings)
{
    for (size_t i = 0; i < size; i++)
    {
        allocator->Free(allocator, strings[i]);
    }

    allocator->Free(allocator, (strings));
}

static char *createFloatTensorWithDataAsOrtValue(const OrtApi *g_ort, OrtMemoryInfo *memory_info, float *input, size_t input_len, int64_t *shape, size_t shape_len, OrtValue **input_tensor)
{
    const size_t model_input_len = input_len * sizeof(float);

    ORT_RETURN_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input, model_input_len, shape, shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, input_tensor));

    return NULL;
}

static char *run(const OrtApi *g_ort, OrtSession *session, OrtMemoryInfo *memory_info, OrtAllocator *allocator,
                 char **input_names, size_t input_names_len, OrtValue *input_value,
                 char **output_names, size_t output_names_len, OrtValue **outputs)
{
    // OrtValue **output_tensor = (OrtValue **)calloc(output_names_len, sizeof(OrtValue *));
    //  allocate and set values separately
    ORT_RETURN_ON_ERROR(g_ort->Run(session, NULL, (const char *const *)input_names, (const OrtValue *const *)&input_value, input_names_len, (const char *const *)output_names, output_names_len, outputs));

    for (size_t i = 0; i < output_names_len; i++)
    {

        printf("\n%ld\n", i);

        int is_tensort = 0;
        ORT_RETURN_ON_ERROR(g_ort->IsTensor(outputs[i], &is_tensort));
        // printf("is tensor %d %ld\n", is_tensort, i);

        if (is_tensort == 1)
        {
            int64_t *output_data = NULL;
            ORT_RETURN_ON_ERROR(g_ort->GetTensorMutableData(outputs[i], (void *)&output_data));
            printf("value %ld\n", output_data[0]);
        }
        else
        {
            OrtValue *inner_value = NULL;
            ORT_RETURN_ON_ERROR(g_ort->GetValue(outputs[i], 0, allocator, &inner_value));

            size_t value_count;
            ORT_RETURN_ON_ERROR(g_ort->GetValueCount(inner_value, &value_count));
            printf("value count %ld\n", value_count);

            for (size_t j = 0; j < value_count; j++)
            {
                OrtValue *map = NULL;
                ORT_RETURN_ON_ERROR(g_ort->GetValue(inner_value, j, allocator, &map));

                int is_tensort = 0;
                ORT_RETURN_ON_ERROR(g_ort->IsTensor(map, &is_tensort));
                printf("\tis tensor %d\n", is_tensort);

                OrtTensorTypeAndShapeInfo *tts;
                ORT_RETURN_ON_ERROR(g_ort->GetTensorTypeAndShape(map, &tts));

                ONNXTensorElementDataType elem_type;
                ORT_RETURN_ON_ERROR(g_ort->GetTensorElementType(tts, &elem_type));

                printf("\telem type %d\n", elem_type);
                switch (elem_type)
                {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                {
                    int64_t *output_data = NULL;
                    ORT_RETURN_ON_ERROR(g_ort->GetTensorMutableData(map, (void *)&output_data));
                    printf("\t\tvalue %ld\n", output_data[0]);
                    printf("\t\tvalue %ld\n", output_data[1]);
                    printf("\t\tvalue %ld\n", output_data[2]);

                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                {
                    float *output_data = NULL;
                    ORT_RETURN_ON_ERROR(g_ort->GetTensorMutableData(map, (void *)&output_data));
                    printf("\t\tvalue %f\n", output_data[0]);
                    printf("\t\tvalue %f\n", output_data[1]);
                    printf("\t\tvalue %f\n", output_data[2]);

                    break;
                }
                }
            }
        }
    }

    return NULL;
}