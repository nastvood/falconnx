package falconnx

/*
	#include <onnxruntime_c_api.h>
*/
import "C"

type ElementType int

const (
	ElementTypeUndefined ElementType = iota
	ElementTypeFloat32
)

func ElementTypeFromC(t C.enum_ONNXTensorElementDataType) ElementType {
	switch t {
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
		return ElementTypeFloat32
	}

	return ElementTypeUndefined
}

func (t ElementType) String() string {
	switch t {
	case ElementTypeFloat32:
		return "float32"
	}

	return "Undefined"
}
