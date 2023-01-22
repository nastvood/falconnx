package falconnx

/*
	#include <onnxruntime_c_api.h>
*/
import "C"

type TypeInfo struct {
	ortTypeInfo *C.OrtTypeInfo
	ortONNXType uint32

	onnxType OnnxType
}
