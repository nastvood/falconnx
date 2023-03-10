package falconnx

/*
	#include "api.h"
*/
import "C"
import (
	"fmt"
)

type SequenceInfo struct {
	TypeInfo *TypeInfo
}

func (si *SequenceInfo) String() string {
	if si == nil {
		return strNil
	}

	return fmt.Sprintf("sequence<%s>", si.TypeInfo)
}

func createSequenceInfo(info *C.OrtTypeInfo) (*SequenceInfo, error) {
	var ortSequenceTypeInfo *C.OrtSequenceTypeInfo // do not free this value
	errMsg := C.castTypeInfoToSequenceTypeInfo(gAPI.ortAPI, info, &ortSequenceTypeInfo)
	if errMsg != nil {
		return nil, newCStatusErr(errMsg)
	}

	var ortTypeInfo *C.OrtTypeInfo
	errMsg = C.getSequenceElementType(gAPI.ortAPI, ortSequenceTypeInfo, &ortTypeInfo)
	if errMsg != nil {
		C.releaseSequenceTypeInfo(gAPI.ortAPI, ortSequenceTypeInfo)
		return nil, newCStatusErr(errMsg)
	}

	typeInfo, err := createAndReleaseTypeInfo(ortTypeInfo)
	if err != nil {
		C.releaseSequenceTypeInfo(gAPI.ortAPI, ortSequenceTypeInfo)
		C.releaseTypeInfo(gAPI.ortAPI, ortTypeInfo)
		return nil, err
	}

	sequenceInfo := &SequenceInfo{
		TypeInfo: typeInfo,
	}

	return sequenceInfo, nil
}
