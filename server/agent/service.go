package agent

import (
	"bufio"
	"context"
	"crypto/ecdsa"
	"crypto/sha256"
	"encoding/asn1"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/big"
	"net/http"
	"os"
	"strings"
	"sync"

	"github.com/copilot-extensions/rag-extension/copilot"
	"github.com/copilot-extensions/rag-extension/embedding"
	"github.com/qdrant/go-client/qdrant"
)

// Service provides and endpoint for this agent to perform chat completions
type Service struct {
	pubKey *ecdsa.PublicKey

	// Singleton
	datasets     []*embedding.Dataset
	datasetsInit *sync.Once

	qClient *qdrant.Client
}

const TOP_N_DATASETS uint64 = 5

func NewService(pubKey *ecdsa.PublicKey, qClient *qdrant.Client) *Service {
	return &Service{
		pubKey:       pubKey,
		datasetsInit: &sync.Once{},
		qClient:      qClient,
	}
}

func (s *Service) ChatCompletion(w http.ResponseWriter, r *http.Request) {
	sig := r.Header.Get("Github-Public-Key-Signature")

	body, err := io.ReadAll(r.Body)
	if err != nil {
		fmt.Println(fmt.Errorf("failed to read request body: %w", err))
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	// Make sure the payload matches the signature. In this way, you can be sure
	// that an incoming request comes from github
	isValid, err := validPayload(body, sig, s.pubKey)
	if err != nil {
		fmt.Printf("failed to validate payload signature: %v\n", err)
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
	if !isValid {
		http.Error(w, "invalid payload signature", http.StatusUnauthorized)
		return
	}

	apiToken := r.Header.Get("X-GitHub-Token")
	integrationID := r.Header.Get("Copilot-Integration-Id")

	var req *copilot.ChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		fmt.Printf("failed to unmarshal request: %v\n", err)
		w.WriteHeader(http.StatusBadRequest)
		return
	}
	if err := s.generateCompletion(r.Context(), integrationID, apiToken, req, w); err != nil {
		fmt.Printf("failed to execute agent: %v\n", err)
		w.WriteHeader(http.StatusInternalServerError)
	}
}

func (s *Service) generateCompletion(ctx context.Context, integrationID, apiToken string, req *copilot.ChatRequest, w io.Writer) error {
	// Initialize the datasets.  In a real application, these would be generated
	// ahead of time and stored in a database
	var messages []copilot.ChatMessage

	// Create embeddings from user messages
	for i := len(req.Messages) - 1; i >= 0; i-- {
		msg := req.Messages[i]
		if msg.Role != "user" {
			continue
		}

		// Filter empty messages
		if msg.Content == "" {
			continue
		}

		emb, err := embedding.Create(ctx, integrationID, apiToken, msg.Content)
		if err != nil {
			return fmt.Errorf("error creating embedding for user message: %w", err)
		}

		// Load most appropriate dataset
		filenames, err := embedding.FindTopNDatasets(TOP_N_DATASETS, emb, s.qClient)
		if err != nil {
			return fmt.Errorf("error computing best dataset")
		}

		if len(filenames) == 0 {
			break
		}

		var contexts []string

		for _, filename := range filenames {
			fmt.Println("using dataset", filename)
			file, err := os.Open(filename)
			if err != nil {
				return fmt.Errorf("failed to open documents: %w", err)
			}

			fileContents, err := io.ReadAll(file)
			if err != nil {
				return fmt.Errorf("failed to read documents: %w", err)
			}

			contexts = append(contexts, string(fileContents))
		}

		//quick hack to ensure we always are in the limit
		chatMessageContent :=
			"You are a helpful assistant that replies to user messages. You are a Ruby on Rails and React expert. You have extensive knowledge of how designs systems work and best practices, especially the Primer design system.  Use the following context when responding to a message.\n" +
				"Context: " + strings.Join(contexts, "\n")[:16384]

		messages = append(messages, copilot.ChatMessage{
			Role:    "system",
			Content: chatMessageContent,
		})

		break
	}

	messages = append(messages, req.Messages...)

	chatReq := &copilot.ChatCompletionsRequest{
		Model:    copilot.ModelGPT35,
		Messages: messages,
		Stream:   true,
	}

	stream, err := copilot.ChatCompletions(ctx, "copilot-chat", apiToken, chatReq)
	if err != nil {
		return fmt.Errorf("failed to get chat completions stream: %w", err)
	}
	defer stream.Close()

	reader := bufio.NewScanner(stream)
	for reader.Scan() {
		buf := reader.Bytes()
		_, err := w.Write(buf)
		if err != nil {
			return fmt.Errorf("failed to write to stream: %w", err)
		}

		if _, err := w.Write([]byte("\n")); err != nil {
			return fmt.Errorf("failed to write delimiter to stream: %w", err)
		}
	}

	if err := reader.Err(); err != nil {
		if errors.Is(err, io.EOF) {
			return nil
		}

		return fmt.Errorf("failed to read from stream: %w", err)
	}

	return nil
}

// asn1Signature is a struct for ASN.1 serializing/parsing signatures.
type asn1Signature struct {
	R *big.Int
	S *big.Int
}

func validPayload(data []byte, sig string, publicKey *ecdsa.PublicKey) (bool, error) {
	asnSig, err := base64.StdEncoding.DecodeString(sig)
	parsedSig := asn1Signature{}
	if err != nil {
		return false, err
	}
	rest, err := asn1.Unmarshal(asnSig, &parsedSig)
	if err != nil || len(rest) != 0 {
		return false, err
	}

	// Verify the SHA256 encoded payload against the signature with GitHub's Key
	digest := sha256.Sum256(data)
	return ecdsa.Verify(publicKey, digest[:], parsedSig.R, parsedSig.S), nil
}
