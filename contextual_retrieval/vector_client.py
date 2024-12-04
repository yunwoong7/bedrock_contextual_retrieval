# vector_client.py
# Description: Vector store client using ChromaDB for document storage and retrieval.

import chromadb
from contextual_retrieval.config import get_config
from typing import List, Dict, Any
from datetime import datetime, UTC

_config = get_config()

class VectorClient:
    def __init__(self, collection_name: str = None):
        """
        Initialize ChromaDB client and collection.

        Parameters:
        - collection_name (str): Name of the collection to use
        """
        self.client = chromadb.PersistentClient(
            path=_config["chromadb"]["persist_directory"]
        )
        self.collection_name = collection_name or _config["chromadb"]["collection_name"]

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"created_at": datetime.now(UTC).isoformat()}
        )

    def add_documents(self,
                      documents: List[str],
                      embeddings: List[List[float]],
                      contexts: List[str] = None,
                      ids: List[str] = None,
                      metadata: List[Dict[str, Any]] = None):
        """
        Add documents with their embeddings to the vector store.
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        if contexts and len(contexts) != len(documents):
            raise ValueError("Number of contexts must match number of documents")

        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{datetime.now(UTC).timestamp()}_{i}" for i in range(len(documents))]

        # Generate metadata if not provided
        if metadata is None:
            metadata = [{"timestamp": datetime.now(UTC).isoformat()} for _ in documents]

        if contexts:
            for i, context in enumerate(contexts):
                metadata[i]["context"] = context

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadata
        )

    def search(self,
               query_embedding: List[float],
               top_k: int = 5,
               include_metadata: bool = True) -> List[Dict]:
        """
        Search for similar documents using query embedding.

        Parameters:
        - query_embedding (List[float]): Query vector
        - top_k (int): Number of results to return
        - include_metadata (bool): Whether to include metadata in results

        Returns:
        - List[Dict]: Search results with scores and metadata
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )

        # Format results
        formatted_results = []
        for doc, meta, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            result = {
                'content': doc,
                'score': 1 - distance,  # Convert distance to similarity score
            }
            if include_metadata:
                result['metadata'] = meta
            formatted_results.append(result)

        return formatted_results

    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            "name": self.collection_name,
            "total_documents": count
        }

    def get_all_documents(self) -> List[Dict]:
        """
        컬렉션의 모든 문서와 메타데이터 반환
        """
        results = self.collection.get(
            include=['documents', 'metadatas']
        )

        formatted_docs = []
        for doc, meta in zip(results['documents'], results['metadatas']):
            formatted_docs.append({
                'content': doc,
                'metadata': meta
            })

        return formatted_docs


if __name__ == "__main__":
    from contextual_retrieval.embedding_models import get_embedding_model


    def run_tests():
        print("\n=== Running Vector Store Tests ===\n")

        try:
            # Initialize clients
            test_collection = f"test_collection_{int(datetime.now(UTC).timestamp())}"
            vector_store = VectorClient(test_collection)
            embedding_model = get_embedding_model()

            print("✓ Vector store initialization successful")
            print(f"  Using collection: {test_collection}")

            # Test documents
            test_documents = [
                "인공지능은 인간의 학습능력과 추론능력을 컴퓨터로 구현한 기술입니다.",
                "머신러닝은 데이터를 기반으로 패턴을 학습하고 예측하는 기술입니다.",
                "딥러닝은 인공신경망을 기반으로 하는 머신러닝의 한 분야입니다."
            ]

            # Generate embeddings
            print("\n생성중인 임베딩...")
            embeddings = [
                embedding_model.encode_single(doc).tolist()
                for doc in test_documents
            ]
            print(f"✓ 임베딩 생성 완료 (dimension: {len(embeddings[0])})")

            # Generate contexts
            contexts = [
                f"문서 {i + 1}에 대한 컨텍스트"
                for i in range(len(test_documents))
            ]

            # Generate custom IDs
            custom_ids = [f"test_doc_{i}" for i in range(len(test_documents))]

            # Add documents
            print("\n문서 추가중...")
            vector_store.add_documents(
                documents=test_documents,
                embeddings=embeddings,
                contexts=contexts,
                ids=custom_ids
            )

            print("✓ Documents added successfully")
            for i, (doc_id, doc) in enumerate(zip(custom_ids, test_documents)):
                print(f"\n  Document #{i + 1}:")
                print(f"    ID: {doc_id}")
                print(f"    Content: {doc[:50]}...")

            # Test search
            print("\n검색 테스트 실행중...")
            test_query = "인공지능의 개념이 궁금합니다."
            print(f"Query: {test_query}")

            query_embedding = embedding_model.encode_single(test_query).tolist()
            results = vector_store.search(query_embedding, top_k=2)

            print("\n✓ Search successful")
            print(f"  Found {len(results)} results")

            for i, result in enumerate(results, 1):
                print(f"\n  Result #{i}:")
                print(f"    Score: {result['score']:.4f}")
                print(f"    Content: {result['content']}")
                print(f"    Context: {result['metadata'].get('context', 'N/A')}")

            # Get collection stats
            stats = vector_store.get_collection_stats()
            print("\n✓ Collection stats:")
            print(f"  Total documents: {stats['total_documents']}")

            # Cleanup
            print("\n정리중...")
            vector_store.client.delete_collection(test_collection)
            print("✓ Test collection cleaned up")

            print("\n✓ All tests passed successfully!")

        except Exception as e:
            print(f"\n✘ Test failed: {str(e)}")
            try:
                vector_store.client.delete_collection(test_collection)
                print("\n✓ Test collection cleaned up")
            except:
                pass
            raise e


    run_tests()