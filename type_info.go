package falconnx

/*
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

	SequenceInfo struct {
		ortSequenceTypeInfo *C.OrtSequenceTypeInfo

		TypeInfo *TypeInfo
	}

	MapInfo struct {
		ortMapTypeInfo *C.OrtMapTypeInfo

		KeyElementType ElementType
		ValueTypeInfo  *TypeInfo
	}

	TypeInfo struct {
		ortTypeInfo *C.OrtTypeInfo
		ortONNXType C.enum_ONNXType

		Type         OnnxType
		TensorInfo   *TensorInfo
		SequenceInfo *SequenceInfo
		MapInfo      *MapInfo
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

func (si *SequenceInfo) release() {
	if si == nil {
		return
	}

	if si.ortSequenceTypeInfo != nil {
		C.releaseSequenceTypeInfo(gApi.ortApi, si.ortSequenceTypeInfo)
	}
}

func (mi *MapInfo) release() {
	if mi == nil {
		return
	}

	if mi.ortMapTypeInfo != nil {
		C.releaseMapTypeInfo(gApi.ortApi, mi.ortMapTypeInfo)
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
	var mapInfo *MapInfo
	var sequenceInfo *SequenceInfo

	var err error
	switch onnxType {
	case OnnxTypeTensor:
		tensorInfo, err = createTensorInfo(info)
		if err != nil {
			return nil, err
		}
	case OnnxTypeSequence:
		sequenceInfo, err = createSequenceInfo(info)
		if err != nil {
			return nil, err
		}
	case OnnxTypeMap:
		mapInfo, err = createMapInfo(info)
		if err != nil {
			return nil, err
		}
	}

	typeInfo := &TypeInfo{
		ortTypeInfo: info,
		ortONNXType: ortONNXType,

		Type:         onnxType,
		TensorInfo:   tensorInfo,
		SequenceInfo: sequenceInfo,
		MapInfo:      mapInfo,
	}

	runtime.SetFinalizer(typeInfo, func(ti *TypeInfo) { ti.release() })

	//fmt.Printf("%s\n", typeInfo.String())

	return typeInfo, nil
}

func (t *TypeInfo) String() string {
	if t == nil {
		return "nil"
	}

	if t.TensorInfo != nil {
		return fmt.Sprintf("%s", t.TensorInfo)
	}

	if t.SequenceInfo != nil {
		return fmt.Sprintf("%s", t.SequenceInfo)
	}

	if t.MapInfo != nil {
		return fmt.Sprintf("%s", t.MapInfo)
	}

	return ""
}

func (t *TensorInfo) String() string {
	if t == nil {
		return "nil"
	}

	return fmt.Sprintf("%s%v", t.ElementType, t.Dimensions)
}

func (s *SequenceInfo) String() string {
	if s == nil {
		return "nil"
	}

	return fmt.Sprintf("sequence<%s>", s.TypeInfo)
}

func (m *MapInfo) String() string {
	if m == nil {
		return "nil"
	}

	return fmt.Sprintf("map<%s,%s>", m.KeyElementType, m.ValueTypeInfo)
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
	if dimensionsCount > 0 {
		errMsg = C.getDimensions(gApi.ortApi, ortTensorTypeAndShapeInfo, (*C.int64_t)(&dimensions[0]), dimensionsCount)
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

func createSequenceInfo(info *C.OrtTypeInfo) (*SequenceInfo, error) {
	var ortSequenceTypeInfo *C.OrtSequenceTypeInfo
	errMsg := C.castTypeInfoToSequenceTypeInfo(gApi.ortApi, info, &ortSequenceTypeInfo)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	var ortTypeInfo *C.OrtTypeInfo
	errMsg = C.getSequenceElementType(gApi.ortApi, ortSequenceTypeInfo, &ortTypeInfo)
	if errMsg != nil {
		C.releaseSequenceTypeInfo(gApi.ortApi, ortSequenceTypeInfo)
		return nil, newCStatusErr(errMsg)
	}

	typeInfo, err := createTypeInfo(ortTypeInfo)
	if err != nil {
		C.releaseSequenceTypeInfo(gApi.ortApi, ortSequenceTypeInfo)
		C.releaseTypeInfo(gApi.ortApi, ortTypeInfo)
		return nil, err
	}

	sequenceInfo := &SequenceInfo{
		ortSequenceTypeInfo: ortSequenceTypeInfo,
		TypeInfo:            typeInfo,
	}

	runtime.SetFinalizer(sequenceInfo, func(si *SequenceInfo) {
		si.release()
	})

	return sequenceInfo, nil
}

func createMapInfo(info *C.OrtTypeInfo) (*MapInfo, error) {
	var ortMapTypeInfo *C.OrtMapTypeInfo
	errMsg := C.castTypeInfoToMapTypeInfo(gApi.ortApi, info, &ortMapTypeInfo)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	var onnxTensorElementDataType C.enum_ONNXTensorElementDataType
	errMsg = C.getMapKeyType(gApi.ortApi, ortMapTypeInfo, &onnxTensorElementDataType)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	var ortTypeInfo *C.OrtTypeInfo
	errMsg = C.getMapValueType(gApi.ortApi, ortMapTypeInfo, &ortTypeInfo)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	valueTypeInfo, err := createTypeInfo(ortTypeInfo)
	if err != nil {
		return nil, err
	}

	mapInfo := &MapInfo{
		ortMapTypeInfo: ortMapTypeInfo,
		KeyElementType: ElementTypeFromC(onnxTensorElementDataType),
		ValueTypeInfo:  valueTypeInfo,
	}

	return mapInfo, nil
}
