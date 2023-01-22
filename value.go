package falconnx

/*
	#include <onnxruntime_c_api.h>
*/
import "C"
import "runtime"

type Value struct {
	ortValue *C.OrtValue
}

func CreateFloatTensor(input []float64) (*Value, error) {
	val, err := createFloatTensor(input)
	if err == nil {
		runtime.SetFinalizer(val, func(val *Value) { val.Release() })
	}

	return createFloatTensor(input)
}

func (v *Value) Release() {
	if v != nil {
		releaseValue(v.ortValue)
	}
}
