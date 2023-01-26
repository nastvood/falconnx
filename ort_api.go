package falconnx

/*
	#cgo CFLAGS: -I${SRCDIR}/onnxruntime/include -Wreturn-type
	#cgo linux LDFLAGS: -L${SRCDIR}/onnxruntime/linux_x64 -lonnxruntime
	#cgo arm64 LDFLAGS: -L${SRCDIR}/onnxruntime/osx_arm64 -lonnxruntime

	#include "ort_api.c"
*/
import "C"
import (
	"errors"
	"unsafe"
)

func allocatorGoStrings(alloocator *C.OrtAllocator, argc C.size_t, argv **C.char) []string {
	length := int(argc)
	tmpslice := (*[1 << 30]*C.char)(unsafe.Pointer(argv))[:length:length]
	gostrings := make([]string, length)
	for i, s := range tmpslice {
		gostrings[i] = C.GoString(s)
	}

	C.releaseAllocatorArrayOfString(alloocator, argc, argv)

	return gostrings
}

func createFloatTensor(input []float32, shape []int64) (*Value, error) {
	if len(input) == 0 {
		return nil, errors.New("the input is empty")
	}

	var ortValue *C.OrtValue = nil
	errMsg := C.createFloatTensorWithDataAsOrtValue(gApi.ortApi, gApi.ortMemoryInfo, (*C.float)(&input[0]), C.ulong(len(input)), (*C.int64_t)(&shape[0]), C.size_t(len(shape)), &ortValue)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	return &Value{
		ortValue: ortValue,
	}, nil
}

// LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/src/onnxruntime/lib && export LD_LIBRARY_PATH
