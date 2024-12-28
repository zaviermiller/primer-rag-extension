package main

/*
This command is used to generate our embeddings and store them in Qdrant for use by our extension

If this were going to production, it would have the ability to:
1. pull from the primer docs repo periodically to get the most up-to-date info
2. actually be able to intelligently handle upserting embeddings (i.e. if a doc is updated, we should update the embeddings, if deleted we should delete the embeddings, etc)
*/

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"

	"github.com/copilot-extensions/rag-extension/config"
	"github.com/copilot-extensions/rag-extension/embedding"
	"github.com/google/uuid"
	"github.com/joho/godotenv"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/qdrant/go-client/qdrant"
)

func main() {
	if err := run(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func run() error {
	if err := godotenv.Load("../.env"); err != nil {
		return fmt.Errorf("error loading .env file: %w", err)
	}

	// override docker host
	os.Setenv("QDRANT_HOST", "localhost")

	config, err := config.New()
	if err != nil {
		return fmt.Errorf("error fetching config: %w", err)
	}

	qClient, err := embedding.Init(config.QdrantHost, config.QdrantPort)
	if err != nil {
		return fmt.Errorf("error creating qdrant client: %w", err)
	}

	defer qClient.Close()

	// create openai client
	oaiClient := openai.NewClient(option.WithAPIKey(config.OpenAIKey))

	// get list of all filenames
	filenames, err := getDataFilenames()
	if err != nil {
		return fmt.Errorf("error getting data filenames: %w", err)
	}

	// create embeddings for each file
	for _, filename := range filenames {
		fmt.Println("processing file:", filename)
		embeddingRes, err := createEmbeddingsForFile(oaiClient, filename)
		if err != nil {
			return fmt.Errorf("error creating embeddings for file: %w", err)
		}

		// create qdrant points from embeddings
		points, err := createQdrantPointsFromEmbeddings(embeddingRes, filename)
		if err != nil {
			return fmt.Errorf("error creating qdrant points from embeddings: %w", err)
		}

		// upsert points into qdrant
		_, err = qClient.Upsert(context.Background(), &qdrant.UpsertPoints{
			CollectionName: embedding.COLLECTION_NAME,
			Points:         points,
		})
		if err != nil {
			return fmt.Errorf("error upserting points: %w", err)
		}
	}

	return nil
}

func getDataFilenames() ([]string, error) {
	var filenames []string
	err := filepath.WalkDir("data", func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if !d.IsDir() && filepath.Ext(path) == ".mdx" {
			filenames = append(filenames, path)
		}
		return nil
	})
	if err != nil {
		err = fmt.Errorf("error walking through \"data\" directory: %w", err)
		return nil, err
	}

	return filenames, nil
}

func createEmbeddingsForFile(oaiClient *openai.Client, filename string) (*openai.CreateEmbeddingResponse, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("error reading in file %s: %w", filename, err)
	}

	fileContent, err := os.ReadFile(file.Name())
	if err != nil {
		return nil, fmt.Errorf("error reading file content: %w", err)
	}

	return oaiClient.Embeddings.New(context.TODO(), openai.EmbeddingNewParams{
		Input: openai.F(openai.EmbeddingNewParamsInputUnion(openai.EmbeddingNewParamsInputArrayOfStrings{string(fileContent)})),
		Model: openai.F(openai.EmbeddingModelTextEmbeddingAda002),
	})
}

func createQdrantPointsFromEmbeddings(embeddingRes *openai.CreateEmbeddingResponse, filename string) ([]*qdrant.PointStruct, error) {
	points := make([]*qdrant.PointStruct, len(embeddingRes.Data))

	for i, emb := range embeddingRes.Data {
		// convert embeddings to float32
		features32 := make([]float32, len(emb.Embedding))
		for i, f64 := range emb.Embedding {
			features32[i] = float32(f64)
		}

		// create qdrant point
		points[i] = &qdrant.PointStruct{
			Id:      qdrant.NewID(uuid.NewString()),
			Vectors: qdrant.NewVectorsDense(features32),
			Payload: qdrant.NewValueMap(map[string]any{
				"filename": filename,
			}),
		}
	}

	return points, nil
}
