package falconnx

/*
	#cgo CFLAGS: -I${SRCDIR}/onnxruntime/include -Wreturn-type
	#cgo linux LDFLAGS: -L${SRCDIR}/onnxruntime/linux_x64 -lonnxruntime
	#cgo arm64 LDFLAGS: -L${SRCDIR}/onnxruntime/osx_arm64 -lonnxruntime

	#include <onnxruntime_c_api.h>
	#include "api.h"
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
	errMsg := C.createMemoryInfo(ortApi, &ortMemoryInfo)
	if errMsg != nil {
		log.Fatal("init onnx api memory info %s", newCStatusErr(errMsg).Error())
	}

	gApi = api{
		ortApi:        ortApi,
		ortMemoryInfo: ortMemoryInfo,
	}
}

// LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/src/onnxruntime/lib && export LD_LIBRARY_PATH
