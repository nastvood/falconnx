package falconnx

/*
	#include "api.h"
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"
)

type (
	ONNXTypeEl interface {
		float32 | uint8 | int8 | uint16 | int16 | int32 | int64 | float64 | uint32 | uint64
	}

	Value struct {
		ortValue *C.OrtValue

		onnxType *OnnxType
		typeInfo *TypeInfo
	}
)

func createValueByOrt(ortValue *C.OrtValue) *Value {
	if ortValue == nil {
		return nil
	}

	val := &Value{
		ortValue: ortValue,
	}

	runtime.SetFinalizer(val, func(val *Value) {
		val.release()
	})

	return val
}

func (v *Value) release() {
	if v == nil || v.ortValue == nil {
		return
	}

	C.releaseValue(gAPI.ortAPI, v.ortValue)
}

func (v *Value) initCheck() error {
	if v == nil {
		return ErrValueIsNil
	}

	if v.ortValue == nil {
		return ErrValueIsNotCreated
	}

	return nil
}

func (v *Value) GetType() (*OnnxType, error) {
	if err := v.initCheck(); err != nil {
		return nil, err
	}

	if v.onnxType != nil {
		return v.onnxType, nil
	}

	var ortOnnxType C.enum_ONNXType
	errMsg := C.getValueType(gAPI.ortAPI, v.ortValue, &ortOnnxType)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	v.onnxType = ref(OnnxTypeFromC(ortOnnxType))

	return v.onnxType, nil
}

func (v *Value) GetTypeInfo() (*TypeInfo, error) {
	if err := v.initCheck(); err != nil {
		return nil, err
	}

	if v.typeInfo != nil {
		return v.typeInfo, nil
	}

	var info *C.OrtTypeInfo
	errMsg := C.getTypeInfo(gAPI.ortAPI, v.ortValue, &info)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	typeInfo, err := createAndReleaseTypeInfo(info)
	if err != nil {
		return nil, err
	}

	//C.releaseTypeInfo(gAPI.ortAPI, info)

	v.typeInfo = typeInfo
	v.onnxType = ref(typeInfo.Type)

	return v.typeInfo, nil
}

func (v *Value) GetValueCount() (int, error) {
	if err := v.initCheck(); err != nil {
		return 0, err
	}

	var size C.size_t
	errMsg := C.getValueCount(gAPI.ortAPI, v.ortValue, &size)
	if errMsg != nil {
		return 0, newCStatusErr(errMsg)
	}

	return int(size), nil
}

// GetValue for only sequence or map.
func (v *Value) GetValue(allocator *allocator, index int) (*Value, error) {
	if err := v.initCheck(); err != nil {
		return nil, err
	}

	onnxType, err := v.GetType()
	if err != nil {
		return nil, err
	}
	if *onnxType != OnnxTypeSequence && *onnxType != OnnxTypeMap {
		return nil, ErrNoSequenceOrMap
	}

	var ortValue *C.OrtValue
	errMsg := C.getValue(gAPI.ortAPI, allocator.getOrtAllocator(), v.ortValue, C.int(index), &ortValue)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	return createValueByOrt(ortValue), nil
}

func CreateTensor[T ONNXTypeEl](input []T, shape []int64) (*Value, error) {
	if len(input) == 0 {
		return nil, ErrSliceIsEmpty
	}

	var t *T

	var typeElement C.ONNXTensorElementDataType
	switch any(t).(type) {
	case *float32:
		typeElement = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
	case *uint8:
		typeElement = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
	case *int8:
		typeElement = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
	case *uint16:
		typeElement = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
	case *int16:
		typeElement = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
	case *int32:
		typeElement = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
	case *int64:
		typeElement = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
	case *float64:
		typeElement = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
	case *uint32:
		typeElement = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
	case *uint64:
		typeElement = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
	default:
		return nil, fmt.Errorf("CreateTensor is not implemented for %T", t)
	}

	data := unsafe.Pointer(&input[0])
	var ortValue *C.OrtValue
	errMsg := C.createTensorWithDataAsOrtValue(gAPI.ortAPI, gAPI.ortMemoryInfo, data, C.ulong(len(input)), (*C.int64_t)(&shape[0]), C.size_t(len(shape)), typeElement, &ortValue)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	val := &Value{
		ortValue: ortValue,
		onnxType: ref(OnnxTypeTensor),
	}

	runtime.SetFinalizer(val, func(val *Value) {
		val.release()
	})

	return val, nil
}

func GetTensorData[T ONNXTypeEl](v *Value, totalElementCount int64) ([]T, error) {
	if err := v.initCheck(); err != nil {
		return nil, err
	}

	onnxType, err := v.GetType()
	if err != nil {
		return nil, err
	}
	if *onnxType != OnnxTypeTensor {
		return nil, ErrNoTensor
	}

	data := unsafe.Pointer(uintptr(0))
	errMsg := C.getTensorMutableData(gAPI.ortAPI, v.ortValue, &data)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	var t *T

	var size int
	switch any(t).(type) {
	case *float32:
		size = C.sizeof_float
	case *uint8:
		size = C.sizeof_uint8_t
	case *int8:
		size = C.sizeof_int8_t
	case *uint16:
		size = C.sizeof_uint16_t
	case *int16:
		size = C.sizeof_int16_t
	case *int32:
		size = C.sizeof_int32_t
	case *int64:
		size = C.sizeof_int64_t
	case *float64:
		size = C.sizeof_double
	case *uint32:
		size = C.sizeof_uint32_t
	case *uint64:
		size = C.sizeof_uint64_t
	default:
		return nil, fmt.Errorf("GetTensorData is not implemented for %T", t)
	}

	count := int(totalElementCount)
	res := make([]T, count)
	for i := 0; i < count; i++ {
		res[i] = *(*T)(unsafe.Add(data, size*i))
	}

	return res, nil
}

func GetMapData[K, V ONNXTypeEl](v *Value, allocator *allocator) (map[K]V, error) {
	if err := v.initCheck(); err != nil {
		return nil, err
	}

	onnxType, err := v.GetType()
	if err != nil {
		return nil, err
	}
	if *onnxType != OnnxTypeMap {
		return nil, ErrNoMap
	}

	const lenValues = 2

	values := make([]*Value, lenValues)
	infs := make([]*TypeInfo, lenValues)
	for i := 0; i < lenValues; i++ {
		val, err := v.GetValue(allocator, i)
		if err != nil {
			return nil, err
		}

		values[i] = val

		info, err := val.GetTypeInfo()
		if err != nil {
			return nil, err
		}

		infs[i] = info
	}

	keys, _ := GetTensorData[K](values[0], infs[0].TensorInfo.TotalElementCount)
	vals, _ := GetTensorData[V](values[1], infs[1].TensorInfo.TotalElementCount)

	res := make(map[K]V, len(keys))
	for i := range keys {
		res[keys[i]] = vals[i]
	}

	return res, nil
}

func GetSeqMapData[K, V ONNXTypeEl](v *Value, allocator *allocator, index int) (map[K]V, error) {
	if err := v.initCheck(); err != nil {
		return nil, err
	}

	onnxType, err := v.GetType()
	if err != nil {
		return nil, err
	}
	if *onnxType != OnnxTypeSequence {
		return nil, ErrNoSequence
	}

	mapValue, err := v.GetValue(allocator, index)
	if err != nil {
		return nil, err
	}

	return GetMapData[K, V](mapValue, allocator)
}
