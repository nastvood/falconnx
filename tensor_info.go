package falconnx

/*
	#include "api.h"
*/
import "C"
import (
	"fmt"
)

type TensorInfo struct {
	ElementType       ElementType
	TotalElementCount int64
	DimensionsCount   uint64
	Dimensions        []int64
}

func (ti *TensorInfo) String() string {
	if ti == nil {
		return strNil
	}

	return fmt.Sprintf("%s%v", ti.ElementType, ti.Dimensions)
}

func createTensorInfo(info *C.OrtTypeInfo) (*TensorInfo, error) {
	var ortTensorTypeAndShapeInfo *C.OrtTensorTypeAndShapeInfo // do not free this value
	errMsg := C.castTypeInfoToTensorInfo(gAPI.ortAPI, info, &ortTensorTypeAndShapeInfo)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	var onnxTensorElementDataType C.enum_ONNXTensorElementDataType
	errMsg = C.getTensorElementType(gAPI.ortAPI, ortTensorTypeAndShapeInfo, &onnxTensorElementDataType)
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

	totalElementCount := int64(-1)
	if index(dimensions, -1) == -1 {
		var cTotalElementCount C.size_t
		errMsg = C.getTensorShapeElementCount(gAPI.ortAPI, ortTensorTypeAndShapeInfo, &cTotalElementCount)
		if errMsg != nil {
			return nil, newCStatusErr(errMsg)
		}

		totalElementCount = int64(cTotalElementCount)
	}

	tensorInfo := &TensorInfo{
		ElementType:       ElementTypeFromC(onnxTensorElementDataType),
		TotalElementCount: totalElementCount,
		DimensionsCount:   uint64(dimensionsCount),
		Dimensions:        dimensions,
	}

	return tensorInfo, nil
}
