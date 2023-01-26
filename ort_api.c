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