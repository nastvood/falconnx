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
	outputCount       uint64

	InputNames     []string
	InputTypesInfo []*TypeInfo
	OutputNames    []string
}

func (s *Session) release() {
	if s == nil {
		return
	}

	releaseSession(gApi.ortApi, s.ortSessionOptions, s.ortSession, s.ortAllocator)
}

func (s *Session) Run(input *Value) ([]*Value, error) {
	if s == nil {
		return nil, nil
	}

	if len(s.OutputNames) == 0 {
		return nil, nil
	}

	inuputNames, inuputNamesRelease := stringsToCharCharArray(s.InputNames)
	defer inuputNamesRelease()

	outputNames, outputNamesRelease := stringsToCharCharArray(s.OutputNames)
	defer outputNamesRelease()

	outputOrtValues := make([]*C.OrtValue, len(s.OutputNames))
	err := run(
		gApi.ortApi,
		s.ortSession,
		gApi.ortMemoryInfo,
		s.ortAllocator,
		inuputNames,
		C.size_t(s.inputCount),
		input.ortValue,
		outputNames,
		C.size_t(s.outputCount),
		(**C.OrtValue)(&outputOrtValues[0]),
	)
	if err != nil {
		return nil, err
	}

	values := make([]*Value, len(s.OutputNames))
	for i, ortValue := range outputOrtValues {
		values[i] = createByOrtValue(ortValue)
	}

	return values, nil
}

func (s *Session) String() string {
	if s == nil {
		return ""
	}

	return fmt.Sprintf("%+v", *s)
}
