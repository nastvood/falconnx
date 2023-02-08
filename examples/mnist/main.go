package main

import (
	"fmt"
	"image/png"
	"log"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/nastvood/falconnx"
)

const imageDir = "./images"

type mnistData struct {
	File  string
	Class int
}

// https://www.kaggle.com/datasets/jidhumohan/mnist-png
var mnistDataset = []mnistData{
	{File: "10.png", Class: 0},
	{File: "10156.png", Class: 1},
	{File: "10016.png", Class: 2},
	{File: "10052.png", Class: 3},
	{File: "1008.png", Class: 4},
	{File: "10065.png", Class: 5},
	{File: "10163.png", Class: 6},
	{File: "10014.png", Class: 7},
	{File: "10124.png", Class: 8},
	{File: "10112.png", Class: 9},
}

func readData(imageName string) ([]float32, error) {
	imagePath := filepath.Join(imageDir, imageName)
	file, err := os.Open(imagePath)
	if err != nil {
		return nil, err
	}

	defer file.Close()

	img, err := png.Decode(file)
	if err != nil {
		return nil, err
	}

	bounds := img.Bounds()
	mul := float32(255. / 65535.)
	res := make([]float32, 0, 28*28)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, _, _, _ := img.At(x, y).RGBA()
			res = append(res, mul*float32(r))
		}
	}

	return res, nil
}

func maxIndex(input []float32) int {
	if len(input) == 0 {
		return -1
	}

	maxIndex := 0
	for i := range input[1:] {
		if input[i+1] > input[maxIndex] {
			maxIndex = i + 1
		}
	}

	return maxIndex
}

func softmax(input []float32) []float32 {
	m := math.Inf(-1)
	for i := range input {
		if m < float64(input[i]) {
			m = float64(input[i])
		}
	}

	exps := make([]float64, len(input))
	var sum float64
	for i := range exps {
		sum += math.Exp(float64(input[i]) - m)
	}

	res := make([]float32, len(input))
	for i := range exps {
		res[i] = float32(math.Exp(float64(input[i]) - m))
	}

	return res
}

func run(session *falconnx.Session, input []float32) ([]float32, error) {
	inputTensor, err := falconnx.CreateTensor(input, session.InputTypesInfo[0].TensorInfo.Dimensions)
	if err != nil {
		return nil, err
	}

	outputs, err := session.Run(inputTensor)
	if err != nil {
		return nil, err
	}

	labels, err := falconnx.GetTensorData[float32](outputs[0], session.OutputTypesInfo[0].TensorInfo.TotalElementCount)
	if err != nil {
		return nil, err
	}

	return labels, nil
}

func process() {
	env, err := falconnx.CreateEnv(falconnx.LoggingLevelWarning, "mnist")
	if err != nil {
		log.Fatalf("create env: %v", err)
	}
	defer env.Release()

	session, err := env.CreateSession("mnist.onnx")
	if err != nil {
		log.Fatalf("create session: %v", err)
	}
	defer session.Release()

	log.Printf("session %s\n", session.String())

	for i, info := range session.InputTypesInfo {
		log.Printf("input[%d]: %s\n", i, info.String())
	}

	for i, info := range session.OutputTypesInfo {
		log.Printf("output[%d]: %s\n", i, info.String())
	}

	for _, data := range mnistDataset {
		input, err := readData(data.File)
		if err != nil {
			log.Fatalf("readData %v: %v", data, err)
		}

		t0 := time.Now()
		res, err := run(session, input)
		if err != nil {
			log.Fatalf("run %v: %v", data, err)
		}
		t1 := time.Now()

		probabilities := softmax(res)
		class := maxIndex(probabilities)

		if data.Class != class {
			log.Printf("want %d, got %d", data.Class, class)
		}

		log.Printf("%+v, probabilities %v, calculated class %d, %s\n", data, probabilities, class, t1.Sub(t0).String())
	}
}

func main() {
	fmt.Println("------------------=== MNIST Start... ===------------------")

	process()

	fmt.Printf("------------------=== MNIST END ===-----------------\n\n")
}
