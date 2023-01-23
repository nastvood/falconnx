package main

import (
	"fmt"
	"log"
	"runtime"
	"time"

	"github.com/nastvood/falconnx"
)

func run(session *falconnx.Session, input []float32) error {
	info := session.InputTypesInfo[0].TensorInfo

	inputTensor, err := falconnx.CreateFloatTensor(input, info.Dimensions)
	if err != nil {
		return err
	}

	outputs, err := session.Run(inputTensor)
	if err != nil {
		return err
	}

	_, err = outputs[0].GetTypeInfo()
	if err != nil {
		return err
	}

	_, err = outputs[1].GetTypeInfo()
	if err != nil {
		return err
	}

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
