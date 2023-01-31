package falconnx

/*
	#include "api.h"
*/
import "C"
import (
	"runtime"
	"unsafe"
)

type Env struct {
	ortEnv *C.OrtEnv
}

func CreateEnv(level LoggingLevel, logid string) (*Env, error) {
	var ortEnv *C.OrtEnv
	cLogid := C.CString(logid)
	errMsg := C.createEnv(gAPI.ortAPI, LoggingLevelToC(level), cLogid, &ortEnv)
	C.free(unsafe.Pointer(cLogid))
	if errMsg != nil {
		err := newCStatusErr(errMsg)
		return nil, err
	}

	env := &Env{
		ortEnv: ortEnv,
	}

	runtime.SetFinalizer(env, func(env *Env) {
		env.release()
	})

	return env, nil
}

func (env *Env) release() {
	if env == nil || env.ortEnv == nil {
		return
	}

	C.releaseEnv(gAPI.ortAPI, env.ortEnv)
}

func (env *Env) CreateSession(modelPath string) (*Session, error) {
	if env == nil {
		return nil, nil
	}

	return createSession(env, modelPath)
}
