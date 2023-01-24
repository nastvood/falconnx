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