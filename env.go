package falconnx

/*
	#include <onnxruntime_c_api.h>
*/
import "C"
import (
	"runtime"
)

type Env struct {
	ortEnv *C.OrtEnv
}

func CreateEnv() (*Env, error) {
	env, err := createEnv()
	if err == nil {
		runtime.SetFinalizer(env, func(env *Env) {
			env.release()
		})
	}

	return env, err
}

func (env *Env) release() {
	if env == nil {
		return
	}

	releaseEnv(env)
	env = nil
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
