package falconnx

/*
	#include <onnxruntime_c_api.h>
*/
import "C"

type Value struct {
	ortValue *C.OrtValue
}

func CreateValue() (*Value, error) {
	return createValue()
}

func (v *Value) Release() {
	if v != nil {
		releaseValue(v.ortValue)
	}
}
