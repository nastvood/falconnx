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
		C.releaseSequenceTypeInfo(gApi.ortApi, si.ortSequenceTypeInfo)
	}
}

func (s *SequenceInfo) String() string {
	if s == nil {
		return "nil"
	}

	return fmt.Sprintf("sequence<%s>", s.TypeInfo)
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
