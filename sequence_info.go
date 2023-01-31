package falconnx

/*
	#include "api.h"
*/
import "C"
import (
	"fmt"
	"runtime"
)

type SequenceInfo struct {
	ortSequenceTypeInfo *C.OrtSequenceTypeInfo

	TypeInfo *TypeInfo
}

func (si *SequenceInfo) release() {
	if si == nil {
		return
	}

	if si.ortSequenceTypeInfo != nil {
		C.releaseSequenceTypeInfo(gAPI.ortAPI, si.ortSequenceTypeInfo)
	}
}

func (si *SequenceInfo) String() string {
	if si == nil {
		return strNil
	}

	return fmt.Sprintf("sequence<%s>", si.TypeInfo)
}

func createSequenceInfo(info *C.OrtTypeInfo) (*SequenceInfo, error) {
	var ortSequenceTypeInfo *C.OrtSequenceTypeInfo
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

	typeInfo, err := createTypeInfo(ortTypeInfo)
	if err != nil {
		C.releaseSequenceTypeInfo(gAPI.ortAPI, ortSequenceTypeInfo)
		C.releaseTypeInfo(gAPI.ortAPI, ortTypeInfo)
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
