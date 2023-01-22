package falconnx

/*
	#include <stdlib.h>
*/
import "C"
import (
	"unsafe"
)

var pSize = C.size_t(unsafe.Sizeof(uintptr(0)))
var floatSize = C.size_t(unsafe.Sizeof(float64(0)))

func stringsToCharCharArray(strs []string) (**C.char, func()) {
	lenStrs := len(strs)
	res := C.calloc(C.size_t(lenStrs), pSize)

	// https://stackoverflow.com/questions/48756732/what-does-1-30c-yourtype-do-exactly-in-cgo/48756785#48756785
	// convert the C array to a Go Array so we can index it
	goArr := (*[1<<30 - 1]*C.char)(res)

	for i, str := range strs {
		goArr[i] = C.CString(str)
	}

	release := func() {
		for i := 0; i < lenStrs; i++ {
			C.free(unsafe.Pointer(goArr[i]))
		}

		C.free(res)
	}

	return (**C.char)(res), release
}

func floatsToFloatArray(vals []float64) *C.float {
	lenVals := len(vals)
	res := C.calloc(C.size_t(lenVals), floatSize)

	goArr := (*[1<<30 - 1]C.float)(res)

	for i, v := range vals {
		goArr[i] = C.float(v)
	}

	return (*C.float)(res)
}
