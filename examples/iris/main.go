package main

import (
	"fmt"
	"log"
	"runtime"
	"time"

	"github.com/nastvood/falconnx"
)

func run(session *falconnx.Session, input []float32) error {
	inputTensor, err := falconnx.CreateFloatTensor(input)
	if err != nil {
		return err
	}

	session.Run(inputTensor)

	return nil
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

	run(session, []float32{5.9, 3.0, 5.1, 1.8})
}

func main() {
	process()

	// check finalizers
	runtime.GC()
	time.Sleep(5 * time.Millisecond)
}
