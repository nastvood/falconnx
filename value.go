package falconnx

/*
	#include "api.h"
*/
import "C"
import (
	"runtime"
	"unsafe"
)

type (
	ONNXTypeEl interface {
		int64 | float32
	}

	Value struct {
		ortValue *C.OrtValue

		typeInfo *TypeInfo
	}
)

func CreateTensor[T ONNXTypeEl](input []T, shape []int64) (*Value, error) {
	if len(input) == 0 {
		return nil, ErrSliceIsEmpty
	}

	var t *T

	var typeElement C.ONNXTensorElementDataType
	switch any(t).(type) {
	case *float32:
		typeElement = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
	case *int64:
		typeElement = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
	}

	data := unsafe.Pointer(&input[0])
	var ortValue *C.OrtValue = nil
	errMsg := C.createTensorWithDataAsOrtValue(gApi.ortApi, gApi.ortMemoryInfo, data, C.ulong(len(input)), (*C.int64_t)(&shape[0]), C.size_t(len(shape)), typeElement, &ortValue)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	val := &Value{
		ortValue: ortValue,
	}

	runtime.SetFinalizer(val, func(val *Value) {
		val.release()
	})

	return val, nil
}

func createByOrtValue(ortValue *C.OrtValue) *Value {
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

	C.releaseValue(gApi.ortApi, v.ortValue)
}

func (v *Value) GetTypeInfo() (*TypeInfo, error) {
	if v == nil {
		return nil, ErrValueIsNil
	}

	if v.ortValue == nil {
		return nil, ErrValueIsNotCreated
	}

	if v.typeInfo != nil {
		return v.typeInfo, nil
	}

	var info *C.OrtTypeInfo
	errMsg := C.getTypeInfo(gApi.ortApi, v.ortValue, &info)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	typeInfo, err := createTypeInfo(info)
	if err != nil {
		return nil, err
	}

	v.typeInfo = typeInfo

	return v.typeInfo, nil
}

func (v *Value) GetValueCount() (int, error) {
	var size C.size_t
	errMsg := C.getValueCount(gApi.ortApi, v.ortValue, &size)
	if errMsg != nil {
		return 0, newCStatusErr(errMsg)
	}

	return int(size), nil
}

func (v *Value) GetValue(allocator *Allocator, index int) (*Value, error) {
	var ortValue *C.OrtValue
	errMsg := C.getValue(gApi.ortApi, allocator.GetOrtAllocator(), v.ortValue, C.int(index), &ortValue)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	return createByOrtValue(ortValue), nil
}

func GetTensorData[T ONNXTypeEl](v *Value, typeInfo *TypeInfo) ([]T, error) {
	data := unsafe.Pointer(uintptr(0))
	errMsg := C.getTensorMutableData(gApi.ortApi, v.ortValue, &data)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	var t *T

	var size int
	switch any(t).(type) {
	case *float32:
		size = C.sizeof_float
	case *int64:
		size = C.sizeof_int64_t
	}

	count := int(typeInfo.TensorInfo.TotalElementCount)
	res := make([]T, count)
	for i := 0; i < count; i++ {
		res[i] = *(*T)(unsafe.Add(data, size*i))
	}

	return res, nil
}

func GetMapData[K, V ONNXTypeEl](v *Value, allocator *Allocator) (map[K]V, error) {
	// TODO: check type need map
	values := make([]*Value, 2)
	infs := make([]*TypeInfo, 2)
	for i := 0; i < 2; i++ {
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

	keys, _ := GetTensorData[K](values[0], infs[0])
	vals, _ := GetTensorData[V](values[1], infs[1])

	res := make(map[K]V, len(keys))
	for i := range keys {
		res[keys[i]] = vals[i]
	}

	return res, nil
}

func GetSeqMapData[K, V ONNXTypeEl](v *Value, allocator *Allocator) (map[K]V, error) {
	mapValue, err := v.GetValue(allocator, 0)
	if err != nil {
		return nil, err
	}

	return GetMapData[K, V](mapValue, allocator)
}
