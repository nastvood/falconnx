package falconnx

/*
	#cgo CFLAGS: -I${SRCDIR}/onnxruntime/include -Wreturn-type -Wswitch
	#cgo linux LDFLAGS: -L${SRCDIR}/onnxruntime/linux_x64 -lonnxruntime
	#cgo arm64 LDFLAGS: -L${SRCDIR}/onnxruntime/osx_arm64 -lonnxruntime

	#include <onnxruntime_c_api.h>
	#include "api.h"
*/
import "C"
import (
	"log"
)

type api struct {
	ortAPI        *C.OrtApi
	ortMemoryInfo *C.OrtMemoryInfo

	AvailableProviders []string
}

var gAPI api

func init() {
	ortAPI := C.createApi()

	var ortMemoryInfo *C.OrtMemoryInfo
	errMsg := C.createMemoryInfo(ortAPI, &ortMemoryInfo)
	if errMsg != nil {
		log.Fatalf("init onnx api memory info %s", newCStatusErr(errMsg).Error())
	}

	var length C.int
	var providers **C.char
	errMsg = C.getAvailableProviders(ortAPI, &providers, &length)
	if errMsg != nil {
		log.Fatalf("get available providers %s", newCStatusErr(errMsg).Error())
	}

	availableProviders := goStrings(int(length), providers)

	errMsg = C.releaseAvailableProviders(ortAPI, providers, length)
	if errMsg != nil {
		log.Fatalf("release available providers %s", newCStatusErr(errMsg).Error())
	}

	gAPI = api{
		ortAPI:        ortAPI,
		ortMemoryInfo: ortMemoryInfo,

		AvailableProviders: availableProviders,
	}
}

func AvailableProviders() []string {
	return gAPI.AvailableProviders
}

// LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/src/onnxruntime/lib && export LD_LIBRARY_PATH
