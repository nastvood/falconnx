package falconnx

/*
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
		err := newCStatusErr(errMsg)
		log.Fatal("init onnx api memory info %s", err.Error())
	}

	gApi = api{
		ortApi:        ortApi,
		ortMemoryInfo: ortMemoryInfo,
	}
}
