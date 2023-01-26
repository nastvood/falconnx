package falconnx

/*
	#include "api.h"
*/
import "C"
import (
	"errors"
	"runtime"
	"unsafe"
)

type Value struct {
	ortValue *C.OrtValue

	typeInfo *TypeInfo
}

var ErrNoTensor = errors.New("no tensor")
var ErrNoValue = errors.New("value is nil")

func CreateFloatTensor(input []float32, shape []int64) (*Value, error) {
	val, err := createFloatTensor(input, shape)
	if err == nil {
		runtime.SetFinalizer(val, func(val *Value) { val.release() })
	}

	return val, nil
}

func createByOrtValue(ortValue *C.OrtValue) *Value {
	if ortValue == nil {
		return nil
	}

	return &Value{
		ortValue: ortValue,
	}
}

func (v *Value) GetTypeInfo() (*TypeInfo, error) {
	if v == nil {
		return nil, ErrNoValue
	}

	if v.ortValue == nil {
		return nil, errors.New("value is not inited")
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
	errMsg := C.getValue(gApi.ortApi, allocator.OrtAllocator, v.ortValue, C.int(index), &ortValue)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	return createByOrtValue(ortValue), nil
}

func (v *Value) release() {
	if v != nil && v.ortValue != nil {
		C.releaseValue(gApi.ortApi, v.ortValue)
	}
}

type ONNXTypeEl interface {
	int64 | float32
}

func GetTensorData[T ONNXTypeEl](v *Value, typeInfo *TypeInfo) ([]T, error) {
	c := unsafe.Pointer(uintptr(0))
	errMsg := C.getTensorMutableData(gApi.ortApi, v.ortValue, &c)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	p := unsafe.Pointer(c)

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
		res[i] = *(*T)(unsafe.Add(p, size*i))
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
