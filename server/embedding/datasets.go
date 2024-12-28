package embedding

import (
	"context"
	"fmt"
	"io"
	"os"

	"github.com/copilot-extensions/rag-extension/copilot"
	"github.com/qdrant/go-client/qdrant"
)

const COLLECTION_NAME = "primer-embeddings"

func Init(qdrantHost string, qdrantPort int) (*qdrant.Client, error) {
	// first check if the collection name exists
	client, err := qdrant.NewClient(&qdrant.Config{
		Host: qdrantHost,
		Port: qdrantPort,
	})

	if err != nil {
		return nil, fmt.Errorf("error creating qdrant client: %w", err)
	}

	exists, err := client.CollectionExists(context.Background(), COLLECTION_NAME)
	if err != nil {
		return nil, fmt.Errorf("error checking if collection exists: %w", err)
	}

	if exists {
		return client, nil
	}

	// Create the collection
	err = client.CreateCollection(context.Background(), &qdrant.CreateCollection{
		CollectionName: COLLECTION_NAME,
		VectorsConfig: qdrant.NewVectorsConfig(&qdrant.VectorParams{
			Size:     1536,
			Distance: qdrant.Distance_Cosine,
		}),
	})

	if err != nil {
		return nil, fmt.Errorf("error creating collection: %w", err)
	}

	return client, nil
}

func Create(ctx context.Context, integrationID, apiToken string, content string) ([]float32, error) {
	resp, err := copilot.Embeddings(ctx, integrationID, apiToken, &copilot.EmbeddingsRequest{
		Model: copilot.ModelEmbeddings,
		Input: []string{content},
	})

	if err != nil {
		return nil, fmt.Errorf("error fetching embeddings: %w", err)
	}

	for _, data := range resp.Data {
		return data.Embedding, nil
	}

	return nil, fmt.Errorf("no embeddings found in response")
}

type Dataset struct {
	Embedding []float32
	Filename  string
}

func GenerateDatasets(integrationID, apiToken string, filenames []string) ([]*Dataset, error) {
	datasets := make([]*Dataset, len(filenames))
	for i, filename := range filenames {
		file, err := os.Open(filename)
		if err != nil {
			return nil, fmt.Errorf("error reading in file %s: %w", filename, err)
		}

		fileContent, err := io.ReadAll(file)

		embedding, err := Create(context.Background(), integrationID, apiToken, string(fileContent))
		if err != nil {
			return nil, fmt.Errorf("error creating embedding for file %s: %w", filename, err)
		}

		datasets[i] = &Dataset{
			Embedding: embedding,
			Filename:  filename,
		}
	}

	return datasets, nil
}

func FindTopNDatasets(n uint64, target []float32, client *qdrant.Client) ([]string, error) {
	searchResult, err := client.Query(context.Background(), &qdrant.QueryPoints{
		CollectionName: COLLECTION_NAME,
		Query:          qdrant.NewQueryDense(target),
		WithPayload:    qdrant.NewWithPayload(true),
		WithVectors:    qdrant.NewWithVectors(true),
		Limit:          &n,
	})
	if err != nil {
		return nil, fmt.Errorf("error querying qdrant: %w", err)
	}

	filenames := make([]string, len(searchResult))

	for i, result := range searchResult {
		filenames[i] = result.Payload["filename"].GetStringValue()
	}

	return filenames, nil
}
