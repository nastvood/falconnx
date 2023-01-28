package falconnx

/*
	#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"unsafe"
)

type StatusErr struct {
	msg string
}

func newCStatusErr(cstr *C.char) *StatusErr {
	err := &StatusErr{
		msg: C.GoString(cstr),
	}

	C.free(unsafe.Pointer(cstr))

	return err
}

func (e *StatusErr) Error() string {
	if e == nil {
		return ""
	}

	return e.msg
}

var (
	ErrNoSequenceOrMap   = errors.New("no sequence or map")
	ErrNoTensor          = errors.New("no tensor")
	ErrNoMap             = errors.New("no map")
	ErrNoSequence        = errors.New("no sequence")
	ErrValueIsNil        = errors.New("value is nil")
	ErrValueIsNotCreated = errors.New("value is not created")
	ErrSliceIsEmpty      = errors.New("slice is empty")
)
