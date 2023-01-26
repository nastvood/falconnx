package falconnx

/*
	#include <onnxruntime_c_api.h>
*/
import "C"

type ElementType int

const (
	ElementTypeUndefined ElementType = iota
	ElementTypeFloat32
	ElementTypeUInt8
	ElementTypeInt8
	ElementTypeUInt16
	ElementTypeInt16
	ElementTypeInt32
	ElementTypeInt64
)

func ElementTypeFromC(t C.enum_ONNXTensorElementDataType) ElementType {
	switch t {
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
		return ElementTypeFloat32
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
		return ElementTypeUInt8
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
		return ElementTypeInt8
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
		return ElementTypeUInt16
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
		return ElementTypeInt16
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
		return ElementTypeInt32
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
		return ElementTypeInt64
	}

	return ElementTypeUndefined
}

func (t ElementType) String() string {
	switch t {
	case ElementTypeFloat32:
		return "float32"
	case ElementTypeUInt8:
		return "uint8"
	case ElementTypeInt8:
		return "int8"
	case ElementTypeUInt16:
		return "uint16"
	case ElementTypeInt16:
		return "int16"
	case ElementTypeInt32:
		return "int32"
	case ElementTypeInt64:
		return "int64"
	}

	return "Undefined"
}
