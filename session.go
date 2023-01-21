package falconnx

/*
	#include <onnxruntime_c_api.h>
*/
import "C"

type Session struct {
	ortSession        *C.OrtSession
	ortSessionOptions *C.OrtSessionOptions
	ortAllocator      *C.OrtAllocator
	inputCount        uint64
	inputNames        []string
	outputCount       uint64
	outputNames       []string
}

func (s *Session) Release() {
	if s == nil {
		return
	}

	releaseSession(gApi.ortApi, s.ortSessionOptions, s.ortSession, s.ortAllocator)
}

func (s *Session) Run() error {
	if s == nil {
		return nil
	}

	inuputNames, inuputNamesRelease := stringsToCharCharArray(s.inputNames)
	defer inuputNamesRelease()

	outputNames, outputNamesRelease := stringsToCharCharArray(s.outputNames)
	defer outputNamesRelease()

	val, err := CreateValue()
	if err != nil {
		return err
	}
	defer val.Release()

	run(gApi.ortApi, s.ortSession, gApi.ortMemoryInfo, s.ortAllocator, inuputNames, C.ulong(s.inputCount), val.ortValue, outputNames, C.ulong(s.outputCount))

	return nil
}
