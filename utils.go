package falconnx

/*
	#include <stdlib.h>
	#include "api.h"
*/
import "C"
import (
	"unsafe"
)

const strNil = "nil"

var pSize = C.size_t(unsafe.Sizeof(uintptr(0)))

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

func allocatorGoStrings(alloocator *C.OrtAllocator, argc C.size_t, argv **C.char) []string {
	gostrings := goStrings(int(argc), argv)

	C.releaseAllocatorArrayOfString(alloocator, argc, argv)

	return gostrings
}

func goStrings(length int, argv **C.char) []string {
	tmpslice := (*[1 << 30]*C.char)(unsafe.Pointer(argv))[:length:length]
	gostrings := make([]string, length)
	for i, s := range tmpslice {
		gostrings[i] = C.GoString(s)
	}

	return gostrings
}

func ref[T any](t T) *T {
	return &t
}
