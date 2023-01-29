package main

import (
	"fmt"
	"log"
	"runtime"
	"time"

	"github.com/nastvood/falconnx"
)

type (
	result struct {
		Labels        []int64
		Probabilities map[int64]float32
	}

	iris struct {
		Feature []float32
		Class   int64
	}
)

var (
	irisLabels = map[int64]string{
		0: "Setosa",
		1: "Versicolor",
		2: "Virginica",
	}

	irisDataset = []iris{
		{Feature: []float32{5.1, 3.5, 1.4, .2}, Class: 0},
		{Feature: []float32{4.9, 3, 1.4, .2}, Class: 0},
		{Feature: []float32{5.6, 2.7, 4.2, 1.3}, Class: 1},
		{Feature: []float32{6.1, 2.8, 4, 1.3}, Class: 1},
		{Feature: []float32{5.9, 3.0, 5.1, 1.8}, Class: 2},
		{Feature: []float32{7.9, 3.8, 6.4, 2}, Class: 2},
	}
)

func run(session *falconnx.Session, input []float32) (*result, error) {
	inputTensor, err := falconnx.CreateTensor(input, session.InputTypesInfo[0].TensorInfo.Dimensions)
	if err != nil {
		return nil, err
	}

	outputs, err := session.Run(inputTensor)
	if err != nil {
		return nil, err
	}

	labels, err := falconnx.GetTensorData[int64](outputs[0], &session.OutputTypesInfo[0].TensorInfo.DimensionsCount)
	if err != nil {
		return nil, err
	}

	probabilities, err := falconnx.GetSeqMapData[int64, float32](outputs[1], session.Allocator, 0)
	if err != nil {
		return nil, err
	}

	return &result{
		Labels:        labels,
		Probabilities: probabilities,
	}, nil
}

func process() {
	env, err := falconnx.CreateEnv(falconnx.LoggingLevelWarning, "iris")
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

	for i := range irisDataset {
		t0 := time.Now()
		res, err := run(session, irisDataset[i].Feature)
		if err != nil {
			log.Fatalf("run: %v", err)
		}
		t1 := time.Now()

		irisLabel, ok := irisLabels[res.Labels[0]]
		if !ok {
			log.Fatalf("not found lable for: %d", res.Labels[0])
		}

		wantLabel := irisLabels[irisDataset[i].Class]
		if irisLabel != wantLabel {
			log.Printf("want %s, got %s", wantLabel, irisLabel)
		}

		log.Printf("%+v:%s, %s\n", *res, irisLabel, t1.Sub(t0).String())
	}
}

func main() {
	fmt.Println("------------------=== IRIS Start... ===------------------")

	process()

	fmt.Printf("------------------=== IRIS END ===-----------------\n\n")

	// check finalizers
	runtime.GC()
	time.Sleep(100 * time.Millisecond)
}
