package falconnx

/*
	#cgo CFLAGS: -I${SRCDIR}/onnxruntime/include
	#cgo linux LDFLAGS: -L${SRCDIR}/onnxruntime/linux_x64 -lonnxruntime
	#cgo linux,arm64 LDFLAGS: -L${SRCDIR}/onnxruntime/osx_arm64 -lonnxruntime

	#include "ort_api.c"
*/
import "C"
import (
	"log"
	"unsafe"
)

type api struct {
	ortApi        *C.OrtApi
	ortMemoryInfo *C.OrtMemoryInfo
}

var gApi api

func init() {
	ortApi := C.createApi()

	var ortMemoryInfo *C.OrtMemoryInfo = nil
	errMsg := C.createMemoryInfo(ortApi, &ortMemoryInfo)
	if errMsg != nil {
		err := newCStatusErr(errMsg)
		log.Fatal("init onnx api memory info %s", err.Error())
	}

	gApi = api{
		ortApi:        ortApi,
		ortMemoryInfo: ortMemoryInfo,
	}
}

func createEnv() (*Env, error) {
	var env *C.OrtEnv = nil
	errMsg := C.createEnv(gApi.ortApi, &env)
	if errMsg != nil {
		err := newCStatusErr(errMsg)
		return nil, err
	}

	return (*Env)(env), nil
}

func releaseEnv(ortApi *C.OrtApi, env *Env) {
	C.releaseEnv(ortApi, (*C.OrtEnv)(env))
}

func allocatorGoStrings(alloocator *C.OrtAllocator, argc C.size_t, argv **C.char) []string {
	length := int(argc)
	tmpslice := (*[1 << 30]*C.char)(unsafe.Pointer(argv))[:length:length]
	gostrings := make([]string, length)
	for i, s := range tmpslice {
		gostrings[i] = C.GoString(s)
	}

	C.releaseAllocatorArrayOfString(alloocator, argc, argv)

	return gostrings
}

func createSession(env *Env, modelPath string) (*Session, error) {
	var sessionOptions *C.OrtSessionOptions = nil
	var session *C.OrtSession = nil
	errMsg := C.createSession(gApi.ortApi, (*C.OrtEnv)(env), &sessionOptions, &session, C.CString(modelPath))
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	var allocator *C.OrtAllocator = nil
	errMsg = C.createAllocator(gApi.ortApi, session, gApi.ortMemoryInfo, &allocator)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	var inputCount C.size_t
	errMsg = C.getInputCount(gApi.ortApi, session, &inputCount)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	var pInputNames **C.char = nil
	errMsg = C.getInputNames(gApi.ortApi, session, allocator, inputCount, &pInputNames)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}
	inputNames := allocatorGoStrings(allocator, inputCount, pInputNames)

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

	return &Session{
		ortSession:        session,
		ortSessionOptions: sessionOptions,
		ortAllocator:      allocator,
		inputCount:        uint64(inputCount),
		inputNames:        inputNames,
		outputCount:       uint64(outputCount),
		outputNames:       outputNames,
	}, nil
}

func releaseSession(ortApi *C.OrtApi, sessionOptions *C.OrtSessionOptions, session *C.OrtSession, allocator *C.OrtAllocator) {
	if sessionOptions != nil {
		C.releaseSessionOptions(ortApi, sessionOptions)
	}

	if session != nil {
		C.releaseSession(ortApi, session)
	}

	if allocator != nil {
		C.releaseAllocator(ortApi, allocator)
	}
}

func createValue() (*Value, error) {
	var ortValue *C.OrtValue = nil
	errMsg := C.createTensorWithDataAsOrtValue(gApi.ortApi, gApi.ortMemoryInfo, &ortValue)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	return &Value{
		ortValue: ortValue,
	}, nil
}

func releaseValue(val *C.OrtValue) {
	if val != nil {
		C.releaseValue(gApi.ortApi, val)
	}
}

func run(ortApi *C.OrtApi, session *C.OrtSession, ortMemotyInfo *C.OrtMemoryInfo, ortAllocatoy *C.OrtAllocator,
	inputNames **C.char, inputNamesLen C.size_t, inputValue *C.OrtValue,
	outputNames **C.char, outputNamesLen C.size_t,
) {
	C.run(ortApi, session, ortMemotyInfo, ortAllocatoy, inputNames, inputNamesLen, inputValue, outputNames, outputNamesLen)
}

// LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/src/onnxruntime/lib
// export LD_LIBRARY_PATH
