package falconnx

/*
	#include <onnxruntime_c_api.h>
*/
import "C"

type OnnxType int

const (
	onnxTypeUnknown OnnxType = iota
	onnxTypeTensor
	onnxTypeSequence
	onnxTypeMap
	onnxTypeOpaque
	onnxTypeSparseTensor
	onnxTypeOptional
)

func OnnxTypeFromC(t uint32) OnnxType {
	switch t {
	case C.ONNX_TYPE_TENSOR:
		return onnxTypeTensor
	case C.ONNX_TYPE_SEQUENCE:
		return onnxTypeSequence
	case C.ONNX_TYPE_MAP:
		return onnxTypeMap
	case C.ONNX_TYPE_OPAQUE:
		return onnxTypeOpaque
	case C.ONNX_TYPE_SPARSETENSOR:
		return onnxTypeSparseTensor
	case C.ONNX_TYPE_OPTIONAL:
		return onnxTypeOptional
	}

	return onnxTypeUnknown
}

func (t OnnxType) String() string {
	switch t {
	case onnxTypeTensor:
		return "onnxTypeTensor"
	case onnxTypeSequence:
		return "onnxTypeSequence"
	case onnxTypeMap:
		return "onnxTypeMap"
	case onnxTypeOpaque:
		return "onnxTypeOpaque"
	case onnxTypeSparseTensor:
		return "onnxTypeSparseTensor"
	case onnxTypeOptional:
		return "onnxTypeOptional"
	}

	return "onnxTypeUnknown"
}
