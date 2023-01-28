package falconnx

/*
	#include "api.h"
*/
import "C"
import "runtime"

type Allocator struct {
	ortAllocator *C.OrtAllocator
}

func createAllocator(ortAllocator *C.OrtAllocator) *Allocator {
	a := &Allocator{
		ortAllocator: ortAllocator,
	}

	runtime.SetFinalizer(a, func(a *Allocator) {
		if a != nil && a.ortAllocator != nil {
			C.releaseAllocator(gApi.ortApi, a.ortAllocator)
		}
	})

	return a
}

func (a *Allocator) getOrtAllocator() *C.OrtAllocator {
	return a.ortAllocator
}
