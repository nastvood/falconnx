package falconnx

/*
	#include "api.h"
*/
import "C"
import "fmt"

type (
	Session struct {
		ortSession        *C.OrtSession
		ortSessionOptions *C.OrtSessionOptions
		outputCount       uint64
		inputCount        uint64

		Allocator *Allocator

		InputNames      []string
		InputTypesInfo  []*TypeInfo
		OutputNames     []string
		OutputTypesInfo []*TypeInfo
	}

	Allocator struct {
		OrtAllocator *C.OrtAllocator
	}
)

func createSession(env *Env, modelPath string) (*Session, error) {
	var sessionOptions *C.OrtSessionOptions
	var session *C.OrtSession
	errMsg := C.createSession(gApi.ortApi, env.ortEnv, &sessionOptions, &session, C.CString(modelPath))
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	var allocator *C.OrtAllocator
	errMsg = C.createAllocator(gApi.ortApi, session, gApi.ortMemoryInfo, &allocator)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	var inputCount C.size_t
	errMsg = C.getInputCount(gApi.ortApi, session, &inputCount)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	var pInputNames **C.char
	errMsg = C.getInputNames(gApi.ortApi, session, allocator, inputCount, &pInputNames)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}
	inputNames := allocatorGoStrings(allocator, inputCount, pInputNames)

	inputsInfo := make([]*TypeInfo, inputCount)
	for i := 0; i < int(inputCount); i++ {
		var info *C.OrtTypeInfo
		errMsg = C.getInputInfo(gApi.ortApi, session, C.ulong(i), &info)
		if errMsg != nil {
			return nil, newCStatusErr(errMsg)
		}

		typeInfo, err := createTypeInfo(info)
		if err != nil {
			return nil, err
		}

		inputsInfo[i] = typeInfo
	}

	var outputCount C.size_t
	errMsg = C.getOutputCount(gApi.ortApi, session, &outputCount)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	var pOutputNames **C.char = nil
	errMsg = C.getOutputNames(gApi.ortApi, session, allocator, outputCount, &pOutputNames)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}
	outputNames := allocatorGoStrings(allocator, outputCount, pOutputNames)

	outputInfo := make([]*TypeInfo, outputCount)
	for i := 0; i < int(outputCount); i++ {
		var info *C.OrtTypeInfo
		errMsg = C.getOutputInfo(gApi.ortApi, session, C.ulong(i), &info)
		if errMsg != nil {
			return nil, newCStatusErr(errMsg)
		}

		typeInfo, err := createTypeInfo(info)
		if err != nil {
			return nil, err
		}

		outputInfo[i] = typeInfo
	}

	return &Session{
		ortSession:        session,
		ortSessionOptions: sessionOptions,
		Allocator: &Allocator{
			OrtAllocator: allocator,
		},
		inputCount:      uint64(inputCount),
		InputNames:      inputNames,
		InputTypesInfo:  inputsInfo,
		outputCount:     uint64(outputCount),
		OutputNames:     outputNames,
		OutputTypesInfo: outputInfo,
	}, nil
}

func (s *Session) release() {
	if s == nil {
		return
	}

	if s.ortSessionOptions != nil {
		C.releaseSessionOptions(gApi.ortApi, s.ortSessionOptions)
	}

	if s.Allocator != nil && s.Allocator.OrtAllocator != nil {
		C.releaseAllocator(gApi.ortApi, s.Allocator.OrtAllocator)
	}

	if s.ortSession != nil {
		C.releaseSession(gApi.ortApi, s.ortSession)
	}
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
		s.Allocator.OrtAllocator,
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
