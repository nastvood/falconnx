package falconnx

/*
	#include <onnxruntime_c_api.h>
*/
import "C"

type OnnxType int

const (
	OnnxTypeUnknown OnnxType = iota
	OnnxTypeTensor
	OnnxTypeSequence
	OnnxTypeMap
	OnnxTypeOpaque
	OnnxTypeSparseTensor
	OnnxTypeOptional
)

func OnnxTypeFromC(t C.enum_ONNXType) OnnxType {
	switch t {
	case C.ONNX_TYPE_TENSOR:
		return OnnxTypeTensor
	case C.ONNX_TYPE_SEQUENCE:
		return OnnxTypeSequence
	case C.ONNX_TYPE_MAP:
		return OnnxTypeMap
	case C.ONNX_TYPE_OPAQUE:
		return OnnxTypeOpaque
	case C.ONNX_TYPE_SPARSETENSOR:
		return OnnxTypeSparseTensor
	case C.ONNX_TYPE_OPTIONAL:
		return OnnxTypeOptional
	}

	return OnnxTypeUnknown
}

func (t OnnxType) String() string {
	switch t {
	case OnnxTypeTensor:
		return "Tensor"
	case OnnxTypeSequence:
		return "Sequence"
	case OnnxTypeMap:
		return "Map"
	case OnnxTypeOpaque:
		return "Opaque"
	case OnnxTypeSparseTensor:
		return "SparseTensor"
	case OnnxTypeOptional:
		return "Optional"
	}

	return "Unknown"
}
