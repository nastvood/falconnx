package falconnx

/*
	#include "api.h"
*/
import "C"
import (
	"fmt"
	"runtime"
)

type TensorInfo struct {
	ortTensorTypeAndShapeInfo *C.OrtTensorTypeAndShapeInfo

	ElementType       ElementType
	TotalElementCount uint64
	DimensionsCount   uint64
	Dimensions        []int64
}

func (ti *TensorInfo) release() {
	if ti == nil {
		return
	}

	if ti.ortTensorTypeAndShapeInfo != nil {
		C.releaseTensorTypeInfo(gAPI.ortAPI, ti.ortTensorTypeAndShapeInfo)
	}
}

func (ti *TensorInfo) String() string {
	if ti == nil {
		return strNil
	}

	return fmt.Sprintf("%s%v", ti.ElementType, ti.Dimensions)
}

func createTensorInfo(info *C.OrtTypeInfo) (*TensorInfo, error) {
	var ortTensorTypeAndShapeInfo *C.OrtTensorTypeAndShapeInfo
	errMsg := C.castTypeInfoToTensorInfo(gAPI.ortAPI, info, &ortTensorTypeAndShapeInfo)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	var onnxTensorElementDataType C.enum_ONNXTensorElementDataType
	errMsg = C.getTensorElementType(gAPI.ortAPI, ortTensorTypeAndShapeInfo, &onnxTensorElementDataType)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	var totalElementCount C.size_t
	errMsg = C.getTensorShapeElementCount(gAPI.ortAPI, ortTensorTypeAndShapeInfo, &totalElementCount)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	var dimensionsCount C.size_t
	errMsg = C.getDimensionsCount(gAPI.ortAPI, ortTensorTypeAndShapeInfo, &dimensionsCount)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	dimensions := make([]int64, dimensionsCount)
	if dimensionsCount > 0 {
		errMsg = C.getDimensions(gAPI.ortAPI, ortTensorTypeAndShapeInfo, (*C.int64_t)(&dimensions[0]), dimensionsCount)
		if errMsg != nil {
			return nil, newCStatusErr(errMsg)
		}
	}

	tensorInfo := &TensorInfo{
		ortTensorTypeAndShapeInfo: ortTensorTypeAndShapeInfo,
		ElementType:               ElementTypeFromC(onnxTensorElementDataType),
		TotalElementCount:         uint64(totalElementCount),
		DimensionsCount:           uint64(dimensionsCount),
		Dimensions:                dimensions,
	}

	runtime.SetFinalizer(tensorInfo, func(ti *TensorInfo) {
		ti.release()
	})

	return tensorInfo, nil
}
