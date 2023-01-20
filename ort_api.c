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

static char *createMemoryInfo(const OrtApi *g_ort, OrtMemoryInfo *memory_info)
{
    ORT_RETURN_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

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

static void releaseSession(const OrtApi *g_ort, OrtSession *session)
{
    g_ort->ReleaseSession(session);
}

static void releaseSessionOptions(const OrtApi *g_ort, OrtSessionOptions *sessionOptions)
{
    g_ort->ReleaseSessionOptions(sessionOptions);
}

static char *run(const OrtApi *g_ort, OrtSession *session)
{
    OrtMemoryInfo *memory_info;
    ORT_RETURN_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    OrtAllocator *allocator;
    ORT_RETURN_ON_ERROR(g_ort->CreateAllocator(session, memory_info, &allocator));

    float model_input[] = {5.9f, 3.0f, 5.1f, 1.8f};
    const size_t model_input_len = 4 * sizeof(float);

    const int64_t input_shape[] = {1, 4};
    const size_t input_shape_len = sizeof(input_shape) / sizeof(input_shape[0]);

    OrtValue *input_tensor = NULL;
    ORT_RETURN_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, model_input, model_input_len, input_shape, input_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

    const char *input_namesi[] = {"float_input"};
    const char *output_namesi[] = {"output_label", "output_probability"};

    size_t output_count = 2;

    OrtValue *output_tensor[] = {NULL, NULL};
    ORT_RETURN_ON_ERROR(g_ort->Run(session, NULL, input_namesi, (const OrtValue *const *)&input_tensor, 1, output_namesi, 2, output_tensor));

    for (size_t i = 0; i < output_count; i++)
    {

        printf("\n");

        int is_tensort = 0;
        ORT_RETURN_ON_ERROR(g_ort->IsTensor(output_tensor[i], &is_tensort));
        // printf("is tensor %d %ld\n", is_tensort, i);

        if (is_tensort == 1)
        {
            int64_t *output_data = NULL;
            ORT_RETURN_ON_ERROR(g_ort->GetTensorMutableData(output_tensor[i], (void *)&output_data));
            printf("value %ld\n", output_data[0]);
        }
        else
        {
            OrtValue *inner_value = NULL;
            ORT_RETURN_ON_ERROR(g_ort->GetValue(output_tensor[i], 0, allocator, &inner_value));

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