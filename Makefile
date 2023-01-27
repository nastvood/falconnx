GOPATH:=$(shell go env | grep GOPATH | sed 's/GOPATH=//' | tr -d '"')
GOBIN:=$(GOPATH)/bin

.PHONY: examples
examples:
	cd examples/iris && go run main.go

lint:
	$(GOBIN)/golangci-lint run
