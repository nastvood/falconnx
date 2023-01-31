package falconnx

/*
	#cgo CFLAGS: -I${SRCDIR}/onnxruntime/include -Wreturn-type -Wswitch
	#cgo linux LDFLAGS: -L${SRCDIR}/onnxruntime/linux_x64 -lonnxruntime
	#cgo arm64 LDFLAGS: -L${SRCDIR}/onnxruntime/osx_arm64 -lonnxruntime

	#include <onnxruntime_c_api.h>
	#include "api.h"
*/
import "C"
import "log"

type api struct {
	ortAPI        *C.OrtApi
	ortMemoryInfo *C.OrtMemoryInfo
}

var gAPI api

func init() {
	ortAPI := C.createApi()

	var ortMemoryInfo *C.OrtMemoryInfo
	errMsg := C.createMemoryInfo(ortAPI, &ortMemoryInfo)
	if errMsg != nil {
		log.Fatalf("init onnx api memory info %s", newCStatusErr(errMsg).Error())
	}

	gAPI = api{
		ortAPI:        ortAPI,
		ortMemoryInfo: ortMemoryInfo,
	}
}

// LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/src/onnxruntime/lib && export LD_LIBRARY_PATH
