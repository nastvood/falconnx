package falconnx

/*
	#include "api.h"
*/
import "C"
import (
	"fmt"
)

type MapInfo struct {
	KeyElementType ElementType
	ValueTypeInfo  *TypeInfo
}

func (mi *MapInfo) String() string {
	if mi == nil {
		return strNil
	}

	return fmt.Sprintf("map<%s,%s>", mi.KeyElementType, mi.ValueTypeInfo)
}

func createMapInfo(info *C.OrtTypeInfo) (*MapInfo, error) {
	var ortMapTypeInfo *C.OrtMapTypeInfo
	errMsg := C.castTypeInfoToMapTypeInfo(gAPI.ortAPI, info, &ortMapTypeInfo)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	defer C.releaseMapTypeInfo(gAPI.ortAPI, ortMapTypeInfo)

	var onnxTensorElementDataType C.enum_ONNXTensorElementDataType
	errMsg = C.getMapKeyType(gAPI.ortAPI, ortMapTypeInfo, &onnxTensorElementDataType)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	var ortTypeInfo *C.OrtTypeInfo
	errMsg = C.getMapValueType(gAPI.ortAPI, ortMapTypeInfo, &ortTypeInfo)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	valueTypeInfo, err := createTypeInfo(ortTypeInfo)
	if err != nil {
		return nil, err
	}

	mapInfo := &MapInfo{
		KeyElementType: ElementTypeFromC(onnxTensorElementDataType),
		ValueTypeInfo:  valueTypeInfo,
	}

	return mapInfo, nil
}
