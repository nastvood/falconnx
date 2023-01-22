package main

import (
	"fmt"
	"log"

	"github.com/nastvood/falconnx"
)

func run(session *falconnx.Session, input []float64) error {
	inputTensor, err := falconnx.CreateFloatTensor(input)
	if err != nil {
		return err
	}

	session.Run(inputTensor)

	return nil
}

func main() {
	env, err := falconnx.CreateEnv()
	if err != nil {
		log.Fatalf("create env: %v", err)
	}

	session, err := env.CreateSession("iris.onnx")
	if err != nil {
		log.Fatalf("create session: %v", err)
	}

	fmt.Printf("sessipn %#v\n", session)

	run(session, []float64{5.9, 3.0, 5.1, 1.8})
	//run(session, []float64{5.6, 3.0, 4.1, 1.3})
}
