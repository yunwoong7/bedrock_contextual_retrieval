# embedding_models.py
# Description: Embedding models for text encoding using AWS Bedrock

from abc import ABC, abstractmethod
import boto3
import json
import numpy as np
from typing import List, Union
from contextual_retrieval.config import get_config

_config = get_config()


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Parameters:
        - texts (List[str]): Texts to encode.

        Returns:
        - np.ndarray: Embeddings of the texts.
        """
        pass

    @abstractmethod
    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text into embedding.

        Parameters:
        - text (str): Text to encode.

        Returns:
        - np.ndarray: Embedding of the text.
        """
        pass


class BedrockEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str = None):
        """Initialize Bedrock embedding model"""
        self.model_name = model_name or _config["bedrock"]["embedding"]["model_id"]
        self.client = boto3.client(
            'bedrock-runtime',
            region_name=_config["bedrock"]["region"]
        )

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text using Bedrock"""
        try:
            response = self.client.invoke_model(
                modelId=self.model_name,
                body=json.dumps({
                    "inputText": text
                })
            )
            response_body = json.loads(response.get('body').read())
            embedding = np.array(response_body.get('embedding'))

            # Verify embedding dimension
            if embedding.shape[0] != _config["bedrock"]["embedding"]["dimension"]:
                raise ValueError(
                    f"Unexpected embedding dimension: {embedding.shape[0]}, "
                    f"expected {_config["bedrock"]["embedding"]["dimension"]}"
                )

            return embedding

        except Exception as e:
            raise Exception(f"Error encoding text with Bedrock: {str(e)}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts using Bedrock"""
        if not isinstance(texts, list):
            texts = [texts]

        embeddings = []
        batch_size = _config["bedrock"]["embedding"]["batch_size"]

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = [self.encode_single(text) for text in batch]
            embeddings.extend(batch_embeddings)

        return np.array(embeddings)


def get_embedding_model(model_name: str = None) -> BaseEmbeddingModel:
    """Factory function for embedding model creation"""
    model_name = model_name or _config["bedrock"]["embedding"]["model_id"]
    return BedrockEmbeddingModel(model_name)


# Test code
if __name__ == "__main__":
    def run_tests():
        print("\n=== Running Embedding Model Tests ===\n")

        try:
            # Initialize model
            model = get_embedding_model()
            print("✓ Model initialization successful")

            # Test single text encoding
            text = "This is a test sentence."
            embedding = model.encode_single(text)
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (_config["bedrock"]["embedding"]["dimension"],)
            print("✓ Single text encoding successful")
            print(f"  Shape: {embedding.shape}")

            # Test batch encoding
            texts = [
                "First test sentence.",
                "Second test sentence.",
                "Third test sentence."
            ]
            embeddings = model.encode(texts)
            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape == (len(texts), _config["bedrock"]["embedding"]["dimension"])
            print("✓ Batch text encoding successful")
            print(f"  Shape: {embeddings.shape}")

            # Test error handling
            try:
                model.encode_single("")
                assert False, "Should have failed on empty input"
            except Exception as e:
                print("✓ Empty input handling successful")

            # Test batch size handling
            batch_size = _config["bedrock"]["embedding"]["batch_size"]
            large_texts = [f"Test sentence {i}" for i in range(batch_size + 5)]
            large_embeddings = model.encode(large_texts)
            assert large_embeddings.shape == (len(large_texts), _config["bedrock"]["embedding"]["dimension"])
            print("✓ Large batch handling successful")

            print("\n✓ All tests passed successfully!")

        except Exception as e:
            print(f"\n✘ Test failed: {str(e)}")
            raise e


    run_tests()