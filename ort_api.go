package falconnx

/*
	#cgo CFLAGS: -I${SRCDIR}/onnxruntime/include
	#cgo linux LDFLAGS: -L${SRCDIR}/onnxruntime/x64 -lonnxruntime

	#include "ort_api.c"
*/
import "C"
import "log"

type api struct {
	ortApi        *C.OrtApi
	ortMemoryInfo *C.OrtMemoryInfo
}

var gApi api

func init() {
	ortApi := C.createApi()

	var ortMemoryInfo *C.OrtMemoryInfo = nil
	errMsg := C.createMemoryInfo(ortApi, ortMemoryInfo)
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

func createSession(env *Env, modelPath string) (*Session, error) {
	var sessionOptions *C.OrtSessionOptions = nil
	var session *C.OrtSession = nil
	errMsg := C.createSession(gApi.ortApi, (*C.OrtEnv)(env), &sessionOptions, &session, C.CString(modelPath))
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	return &Session{
		ortSession:        session,
		ortSessionOptions: sessionOptions,
	}, nil
}

func releaseSession(ortApi *C.OrtApi, sessionOptions *C.OrtSessionOptions, session *C.OrtSession) {
	if sessionOptions != nil {
		C.releaseSessionOptions(ortApi, sessionOptions)
	}

	if session != nil {
		C.releaseSession(ortApi, session)
	}
}

func run(ortApi *C.OrtApi, session *C.OrtSession) {
	C.run(ortApi, session)
}

// LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/src/onnxruntime/lib
// export LD_LIBRARY_PATH
