package falconnx

/*
	#include "api.h"
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"
)

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
)

func createSession(env *Env, modelPath string) (*Session, error) {
	var ortSessionOptions *C.OrtSessionOptions
	var ortSession *C.OrtSession
	cModelPath := C.CString(modelPath)
	errMsg := C.createSession(gApi.ortApi, env.ortEnv, &ortSessionOptions, &ortSession, cModelPath)
	C.free(unsafe.Pointer(cModelPath))
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	session := &Session{
		ortSession:        ortSession,
		ortSessionOptions: ortSessionOptions,
	}
	runtime.SetFinalizer(session, func(session *Session) {
		session.release()
	})

	var ortAllocator *C.OrtAllocator
	errMsg = C.createAllocator(gApi.ortApi, ortSession, gApi.ortMemoryInfo, &ortAllocator)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}
	session.Allocator = createAllocator(ortAllocator)

	var inputCount C.size_t
	errMsg = C.getInputCount(gApi.ortApi, ortSession, &inputCount)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	var pInputNames **C.char
	errMsg = C.getInputNames(gApi.ortApi, ortSession, ortAllocator, inputCount, &pInputNames)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}
	inputNames := allocatorGoStrings(ortAllocator, inputCount, pInputNames)

	inputsInfo := make([]*TypeInfo, inputCount)
	for i := 0; i < int(inputCount); i++ {
		var info *C.OrtTypeInfo
		errMsg = C.getInputInfo(gApi.ortApi, ortSession, C.ulong(i), &info)
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
	errMsg = C.getOutputCount(gApi.ortApi, ortSession, &outputCount)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	var pOutputNames **C.char = nil
	errMsg = C.getOutputNames(gApi.ortApi, ortSession, ortAllocator, outputCount, &pOutputNames)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}
	outputNames := allocatorGoStrings(ortAllocator, outputCount, pOutputNames)

	outputInfo := make([]*TypeInfo, outputCount)
	for i := 0; i < int(outputCount); i++ {
		var info *C.OrtTypeInfo
		errMsg = C.getOutputInfo(gApi.ortApi, ortSession, C.ulong(i), &info)
		if errMsg != nil {
			return nil, newCStatusErr(errMsg)
		}

		typeInfo, err := createTypeInfo(info)
		if err != nil {
			return nil, err
		}

		outputInfo[i] = typeInfo
	}

	session.inputCount = uint64(inputCount)
	session.InputNames = inputNames
	session.InputTypesInfo = inputsInfo
	session.outputCount = uint64(outputCount)
	session.OutputNames = outputNames
	session.OutputTypesInfo = outputInfo

	return session, nil
}

func (s *Session) release() {
	if s == nil {
		return
	}

	if s.ortSessionOptions != nil {
		C.releaseSessionOptions(gApi.ortApi, s.ortSessionOptions)
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
	errMsg := C.run(
		gApi.ortApi,
		s.ortSession,
		gApi.ortMemoryInfo,
		s.Allocator.GetOrtAllocator(),
		inuputNames,
		C.size_t(s.inputCount),
		input.ortValue,
		outputNames,
		C.size_t(s.outputCount),
		(**C.OrtValue)(&outputOrtValues[0]),
	)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
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
