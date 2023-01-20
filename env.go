package falconnx

/*
	#include <onnxruntime_c_api.h>
*/
import "C"

type Env C.OrtEnv

func CreateEnv() (*Env, error) {
	return createEnv()
}

func (env *Env) Release() {
	if env == nil {
		return
	}

	releaseEnv(gApi.ortApi, env)
	env = nil
}

func (env *Env) CreateSession(modelPath string) (*Session, error) {
	if env == nil {
		return nil, nil
	}

	return createSession(env, modelPath)
}
