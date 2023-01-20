package falconnx

/*
	#include <onnxruntime_c_api.h>
*/
import "C"

type Session struct {
	ortSession        *C.OrtSession
	ortSessionOptions *C.OrtSessionOptions
}

func (s *Session) Release() {
	if s == nil {
		return
	}

	releaseSession(gApi.ortApi, s.ortSessionOptions, s.ortSession)
}

func (s *Session) Run() {
	if s == nil {
		return
	}

	run(gApi.ortApi, s.ortSession)
}
