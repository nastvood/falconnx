package falconnx

/*
	#include "api.h"
*/
import "C"
import "runtime"

type allocator struct {
	ortAllocator *C.OrtAllocator
}

func createAllocator(ortAllocator *C.OrtAllocator) *allocator {
	a := &allocator{
		ortAllocator: ortAllocator,
	}

	runtime.SetFinalizer(a, func(a *allocator) {
		if a != nil && a.ortAllocator != nil {
			C.releaseAllocator(gApi.ortApi, a.ortAllocator)
		}
	})

	return a
}

func (a *allocator) getOrtAllocator() *C.OrtAllocator {
	return a.ortAllocator
}
