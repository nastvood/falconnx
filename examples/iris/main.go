package main

import (
	"fmt"
	"log"

	"github.com/nastvood/falconnx"
)

func main() {
	env, err := falconnx.CreateEnv()
	if err != nil {
		log.Fatalf("create env: %v", err)
	}
	defer env.Release()

	session, err := env.CreateSession("iris.onnx")
	if err != nil {
		log.Fatalf("create session: %v", err)
	}
	defer session.Release()

	fmt.Printf("sessipn %#v\n", session)

	session.Run()
}
