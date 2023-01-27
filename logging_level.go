package falconnx

/*
	#include <onnxruntime_c_api.h>
*/
import "C"

type LoggingLevel int

const (
	LoggingLevelVerbose LoggingLevel = iota
	LoggingLevelInfo
	LoggingLevelWarning
	LoggingLevelError
	LoggingLevelFatal
)

func LoggingLevelToC(l LoggingLevel) C.OrtLoggingLevel {
	switch l {
	case LoggingLevelInfo:
		return C.ORT_LOGGING_LEVEL_INFO
	case LoggingLevelWarning:
		return C.ORT_LOGGING_LEVEL_WARNING
	case LoggingLevelError:
		return C.ORT_LOGGING_LEVEL_ERROR
	case LoggingLevelFatal:
		return C.ORT_LOGGING_LEVEL_FATAL
	}

	return C.ORT_LOGGING_LEVEL_VERBOSE
}
