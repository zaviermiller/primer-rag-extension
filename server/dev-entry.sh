#!/bin/bash

# download go deps
go mod download

# run the server
go run ./cmd/serve/main.go