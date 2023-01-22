package falconnx

/*
	#include <onnxruntime_c_api.h>
*/
import "C"
import "fmt"

type Session struct {
	ortSession        *C.OrtSession
	ortSessionOptions *C.OrtSessionOptions
	ortAllocator      *C.OrtAllocator
	inputCount        uint64
	inputNames        []string
	inputTypesInfo    []*TypeInfo
	outputCount       uint64
	outputNames       []string
}

func (s *Session) release() {
	if s == nil {
		return
	}

	releaseSession(gApi.ortApi, s.ortSessionOptions, s.ortSession, s.ortAllocator)
}

func (s *Session) Run(input *Value) error {
	if s == nil {
		return nil
	}

	inuputNames, inuputNamesRelease := stringsToCharCharArray(s.inputNames)
	defer inuputNamesRelease()

	outputNames, outputNamesRelease := stringsToCharCharArray(s.outputNames)
	defer outputNamesRelease()

	run(gApi.ortApi, s.ortSession, gApi.ortMemoryInfo, s.ortAllocator, inuputNames, C.ulong(s.inputCount), input.ortValue, outputNames, C.ulong(s.outputCount))

	return nil
}

func (s *Session) String() string {
	if s == nil {
		return ""
	}

	return fmt.Sprintf("%+v", *s)
}
