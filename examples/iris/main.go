package main

import (
	"fmt"
	"log"
	"runtime"
	"time"

	"github.com/nastvood/falconnx"
)

func run(session *falconnx.Session, input []float32) error {
	inputTensor, err := falconnx.CreateFloatTensor(input, session.InputTypesInfo[0].TensorInfo.Dimensions)
	if err != nil {
		return err
	}

	outputs, err := session.Run(inputTensor)
	if err != nil {
		return err
	}

	inputData, _ := falconnx.GetTensorData[float32](inputTensor, session.InputTypesInfo[0])
	fmt.Printf("input data[0] %v\n", inputData)

	labels, _ := falconnx.GetTensorData[int64](outputs[0], session.OutputTypesInfo[0])
	fmt.Printf("labels %v\n", labels)

	mapValue, _ := outputs[1].GetValue(session.Allocator, 0)
	probabilities, _ := falconnx.GetMapData[int64, float32](mapValue, session.Allocator)
	fmt.Printf("probabilities %v\n", probabilities)

	return err
}

func process() {
	env, err := falconnx.CreateEnv()
	if err != nil {
		log.Fatalf("create env: %v", err)
	}

	session, err := env.CreateSession("iris.onnx")
	if err != nil {
		log.Fatalf("create session: %v", err)
	}

	fmt.Printf("session %s\n", session.String())

	for i, info := range session.InputTypesInfo {
		fmt.Printf("input[%d]: %s\n", i, info.String())
	}

	for i, info := range session.OutputTypesInfo {
		fmt.Printf("output[%d]: %s\n", i, info.String())
	}

	err = run(session, []float32{5.9, 3.0, 5.1, 1.8})
	if err != nil {
		log.Fatalf("run: %v", err)
	}
}

func main() {
	process()

	// check finalizers
	runtime.GC()
	time.Sleep(5 * time.Millisecond)
}
