package cmd

import (
	"fmt"

	"github.com/copilot-extensions/rag-extension/config"
	"github.com/copilot-extensions/rag-extension/embedding"
)

func main() {
	config, err := config.New()
	if err != nil {
		fmt.Errorf("error fetching config: %w", err)
		return
	}

	client, err := embedding.Init(config.QdrantHost, config.QdrantPort)
	if err != nil {
		fmt.Errorf("error creating qdrant client: %w", err)
		return
	}

	defer client.Close()

	// ok we need to iterate through, generate our embeddings
}
