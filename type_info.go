package falconnx

/*
	#include "api.h"
*/
import "C"

type TypeInfo struct {
	ortONNXType C.enum_ONNXType

	Type         OnnxType
	TensorInfo   *TensorInfo
	SequenceInfo *SequenceInfo
	MapInfo      *MapInfo
}

func createTypeInfo(info *C.OrtTypeInfo) (*TypeInfo, error) {
	var ortONNXType C.enum_ONNXType
	errMsg := C.getOnnxTypeFromTypeInfo(gAPI.ortAPI, info, &ortONNXType)
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
		ortONNXType: ortONNXType,

		Type:         onnxType,
		TensorInfo:   tensorInfo,
		SequenceInfo: sequenceInfo,
		MapInfo:      mapInfo,
	}

	return typeInfo, nil
}

func (ti *TypeInfo) String() string {
	if ti == nil {
		return strNil
	}

	if ti.TensorInfo != nil {
		return ti.TensorInfo.String()
	}

	if ti.SequenceInfo != nil {
		return ti.SequenceInfo.String()
	}

	if ti.MapInfo != nil {
		return ti.MapInfo.String()
	}

	return ""
}
