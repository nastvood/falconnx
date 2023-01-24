package falconnx

/*
	#include "api.h"
*/
import "C"
import (
	"runtime"
)

type Env struct {
	ortEnv *C.OrtEnv
}

func CreateEnv() (*Env, error) {
	var ortEnv *C.OrtEnv
	errMsg := C.createEnv(gApi.ortApi, &ortEnv)
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
	if env == nil {
		return
	}

	C.releaseEnv(gApi.ortApi, env.ortEnv)
}

func (env *Env) CreateSession(modelPath string) (*Session, error) {
	if env == nil {
		return nil, nil
	}

	session, err := createSession(env, modelPath)
	if err == nil {
		runtime.SetFinalizer(session, func(session *Session) {
			session.release()
		})
	}

	return session, err
}
