package main

import (
	"fmt"
	"log"
	"runtime"
	"time"

	"github.com/nastvood/falconnx"
)

type result struct {
	Labels        []int64
	Probabilities map[int64]float32
}

func run(session *falconnx.Session, input []float32) (*result, error) {
	inputTensor, err := falconnx.CreateFloatTensor(input, session.InputTypesInfo[0].TensorInfo.Dimensions)
	if err != nil {
		return nil, err
	}

	outputs, err := session.Run(inputTensor)
	if err != nil {
		return nil, err
	}

	labels, err := falconnx.GetTensorData[int64](outputs[0], session.OutputTypesInfo[0])
	if err != nil {
		return nil, err
	}

	probabilities, err := falconnx.GetSeqMapData[int64, float32](outputs[1], session.Allocator)
	if err != nil {
		return nil, err
	}

	return &result{
		Labels:        labels,
		Probabilities: probabilities,
	}, nil
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

	t0 := time.Now()
	res, err := run(session, []float32{5.9, 3.0, 5.1, 1.8})
	if err != nil {
		log.Fatalf("run: %v", err)
	}
	t1 := time.Now()

	log.Printf("%+v %s\n", *res, t1.Sub(t0).String())
}

func main() {
	process()

	// check finalizers
	runtime.GC()
	time.Sleep(5 * time.Millisecond)
}
