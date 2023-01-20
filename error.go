package falconnx

/*
	#include <stdlib.h>
*/
import "C"
import "unsafe"

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
