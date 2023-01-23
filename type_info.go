package falconnx

/*
	#include <onnxruntime_c_api.h>
	#include "api.h"
*/
import "C"
import (
	"fmt"
	"runtime"
)

type (
	TensorInfo struct {
		ortTensorTypeAndShapeInfo *C.OrtTensorTypeAndShapeInfo

		ElementType       ElementType
		TotalElementCount uint64
		DimensionsCount   uint64
		Dimensions        []int64
	}

	TypeInfo struct {
		ortTypeInfo         *C.OrtTypeInfo
		ortONNXType         C.enum_ONNXType
		ortMapTypeInfo      *C.OrtMapTypeInfo
		ortSequenceTypeInfo *C.OrtSequenceTypeInfo

		Type       OnnxType
		TensorInfo *TensorInfo
	}
)

func (ti *TypeInfo) release() {
	if ti == nil {
		return
	}

	if ti.ortTypeInfo != nil {
		C.releaseTypeInfo(gApi.ortApi, ti.ortTypeInfo)
	}
}

func (ti *TensorInfo) release() {
	if ti == nil {
		return
	}

	if ti.ortTensorTypeAndShapeInfo != nil {
		C.releaseTensorTypeInfo(gApi.ortApi, ti.ortTensorTypeAndShapeInfo)
	}
}

func createTypeInfo(info *C.OrtTypeInfo) (*TypeInfo, error) {
	var ortONNXType C.enum_ONNXType
	errMsg := C.getOnnxTypeFromTypeInfo(gApi.ortApi, info, &ortONNXType)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	onnxType := OnnxTypeFromC(ortONNXType)

	var tensorInfo *TensorInfo
	var ortMapTypeInfo *C.OrtMapTypeInfo
	var ortSequenceTypeInfo *C.OrtSequenceTypeInfo

	var err error
	switch onnxType {
	case OnnxTypeTensor:
		tensorInfo, err = createTensorInfo(info)
		if err != nil {
			return nil, err
		}
	}

	typeInfo := &TypeInfo{
		ortTypeInfo:         info,
		ortONNXType:         ortONNXType,
		ortMapTypeInfo:      ortMapTypeInfo,
		ortSequenceTypeInfo: ortSequenceTypeInfo,

		Type:       onnxType,
		TensorInfo: tensorInfo,
	}

	runtime.SetFinalizer(typeInfo, func(ti *TypeInfo) { ti.release() })

	fmt.Printf("%s\n", typeInfo.String())

	return typeInfo, nil
}

func (t *TypeInfo) String() string {
	if t == nil {
		return ""
	}

	return fmt.Sprintf("%+v", *t)
}

func (t *TensorInfo) String() string {
	if t == nil {
		return ""
	}

	return fmt.Sprintf("%s%v", t.ElementType, t.Dimensions)
}

func createTensorInfo(info *C.OrtTypeInfo) (*TensorInfo, error) {
	var ortTensorTypeAndShapeInfo *C.OrtTensorTypeAndShapeInfo
	errMsg := C.castTypeInfoToTensorInfo(gApi.ortApi, info, &ortTensorTypeAndShapeInfo)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	var onnxTensorElementDataType C.enum_ONNXTensorElementDataType
	errMsg = C.getTensorElementType(gApi.ortApi, ortTensorTypeAndShapeInfo, &onnxTensorElementDataType)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	var totalElementCount C.size_t
	errMsg = C.getTensorShapeElementCount(gApi.ortApi, ortTensorTypeAndShapeInfo, &totalElementCount)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	var dimensionsCount C.size_t
	errMsg = C.getDimensionsCount(gApi.ortApi, ortTensorTypeAndShapeInfo, &dimensionsCount)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	dimensions := make([]int64, dimensionsCount)
	errMsg = C.getDimensions(gApi.ortApi, ortTensorTypeAndShapeInfo, (*C.int64_t)(&dimensions[0]), dimensionsCount)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
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
