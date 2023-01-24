package falconnx

/*
	#include "api.h"
*/
import "C"
import (
	"errors"
	"runtime"
)

type Value struct {
	ortValue *C.OrtValue

	typeInfo *TypeInfo
}

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
		return nil, errors.New("value is nil")
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

func (v *Value) release() {
	if v != nil {
		releaseValue(v.ortValue)
	}
}

func releaseValue(val *C.OrtValue) {
	if val != nil {
		C.releaseValue(gApi.ortApi, val)
	}
}
