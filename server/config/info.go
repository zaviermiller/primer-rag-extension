package config

import (
	"fmt"
	"os"
	"strconv"
)

type Info struct {
	// Port is the local port on which the application will run
	Port string

	// FQDN (for Fully-Qualified Domain Name) is the internet facing host address
	// where application will live (e.g. https://example.com)
	FQDN string

	// ClientID comes from your configured GitHub app
	ClientID string

	// ClientSecret comes from your configured GitHub app
	ClientSecret string

	// QdrantHost is the host address of the qdrant server
	QdrantHost string

	// QdrantPort is the port of the qdrant server
	QdrantPort int

	// OpenAIKey is the API key for OpenAI
	OpenAIKey string
}

const (
	portEnv         = "PORT"
	clientIdEnv     = "CLIENT_ID"
	clientSecretEnv = "CLIENT_SECRET"
	fqdnEnv         = "FQDN"
	qdrantHostEnv   = "QDRANT_HOST"
	qdrantPortEnv   = "QDRANT_PORT"
	openAiKeyEnv    = "OPENAI_API_KEY"
)

func New() (*Info, error) {

	port := os.Getenv(portEnv)
	if port == "" {
		return nil, fmt.Errorf("%s environment variable required", portEnv)
	}

	fqdn := os.Getenv(fqdnEnv)
	if fqdn == "" {
		return nil, fmt.Errorf("%s environment variable required", fqdnEnv)
	}

	clientID := os.Getenv(clientIdEnv)
	if clientID == "" {
		return nil, fmt.Errorf("%s environment variable required", clientIdEnv)
	}

	clientSecret := os.Getenv(clientSecretEnv)
	if clientSecret == "" {
		return nil, fmt.Errorf("%s environment variable required", clientSecretEnv)
	}

	qdrantHost := os.Getenv(qdrantHostEnv)
	if qdrantHost == "" {
		return nil, fmt.Errorf("%s environment variable required", qdrantHostEnv)
	}

	// Get and parse the port number as int
	qdrantPortStr := os.Getenv(qdrantPortEnv)
	if qdrantPortStr == "" {
		return nil, fmt.Errorf("%s environment variable required", qdrantPortEnv)
	}

	qdrantPort, err := strconv.Atoi(qdrantPortStr)
	if err != nil {
		return nil, fmt.Errorf("error parsing %s: %w", qdrantPortEnv, err)
	}

	openAiKey := os.Getenv(openAiKeyEnv)
	if openAiKey == "" {
		return nil, fmt.Errorf("%s environment variable required", openAiKeyEnv)
	}

	return &Info{
		Port:         port,
		FQDN:         fqdn,
		ClientID:     clientID,
		ClientSecret: clientSecret,
		QdrantHost:   qdrantHost,
		QdrantPort:   qdrantPort,
		OpenAIKey:    openAiKey,
	}, nil
}
