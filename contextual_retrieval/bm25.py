# bm25.py
# Description: BM25 retrieval functionality.

from rank_bm25 import BM25Okapi
from typing import List, Tuple

class BM25Retriever:
   def __init__(self):
       """
       Initialize the BM25 retriever.
       """
       self.bm25 = None
       self.documents = []
       self.tokenized_corpus = []

   def index_chunks(self, chunks: List[str]) -> None:
       """
       Index the chunks using BM25.
       Parameters:
       - chunks (List[str]): The text chunks to index.
       """
       if not chunks:
           raise ValueError("No chunks provided for indexing.")

       self.documents = chunks
       # 간단한 토크나이징 (split)
       self.tokenized_corpus = [doc.lower().split() for doc in chunks]
       self.bm25 = BM25Okapi(self.tokenized_corpus)

   def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
       """
       Retrieve relevant documents using BM25.
       Parameters:
       - query (str): The query text.
       - top_k (int): Number of top results to return.
       Returns:
       - List[Tuple[str, float]]: List of tuples (document, score).
       """
       if not self.bm25:
           raise ValueError("No documents have been indexed. Call index_chunks() first.")

       tokenized_query = query.lower().split()
       scores = self.bm25.get_scores(tokenized_query)
       top_n = scores.argsort()[-top_k:][::-1]
       results = [(self.documents[i], scores[i]) for i in top_n]
       return results

   def get_document_count(self) -> int:
       """
       Get the number of indexed documents.
       Returns:
       - int: The number of indexed documents.
       """
       return len(self.documents)

   def get_average_document_length(self) -> float:
       """
       Get the average length of indexed documents.
       Returns:
       - float: The average document length.
       """
       if not self.tokenized_corpus:
           raise ValueError("No documents have been indexed.")
       return sum(len(doc) for doc in self.tokenized_corpus) / len(self.tokenized_corpus)


if __name__ == "__main__":
   def run_tests():
       print("\n=== Running BM25 Retriever Tests ===\n")

       try:
           # Initialize retriever
           retriever = BM25Retriever()
           print("✓ BM25 Retriever initialized")

           # Test documents
           test_documents = [
               "인공지능은 인간의 학습능력과 추론능력을 컴퓨터로 구현한 기술입니다.",
               "머신러닝은 데이터를 기반으로 패턴을 학습하고 예측하는 기술입니다.",
               "딥러닝은 인공신경망을 기반으로 하는 머신러닝의 한 분야입니다.",
               "자연어 처리는 컴퓨터가 인간의 언어를 이해하고 처리하는 기술입니다.",
               "강화학습은 에이전트가 환경과 상호작용하며 보상을 최대화하는 방향으로 학습하는 방법입니다."
           ]

           # Test indexing
           print("\nIndexing test documents...")
           retriever.index_chunks(test_documents)
           print(f"✓ Documents indexed successfully")
           print(f"  Total documents: {retriever.get_document_count()}")
           print(f"  Average document length: {retriever.get_average_document_length():.2f}")

           # Test retrieval
           test_queries = [
               "인공지능이란 무엇인가요?",
               "머신러닝과 딥러닝의 차이는?",
               "자연어 처리 기술이란?"
           ]

           print("\nTesting retrieval...")
           for query in test_queries:
               print(f"\nQuery: {query}")
               results = retriever.retrieve(query, top_k=2)
               for i, (doc, score) in enumerate(results, 1):
                   print(f"\n  Result #{i}:")
                   print(f"    Score: {score:.4f}")
                   print(f"    Content: {doc}")

           # Test error cases
           print("\nTesting error cases...")

           # Test empty index
           empty_retriever = BM25Retriever()
           try:
               empty_retriever.retrieve("test query")
               print("✘ Should have raised ValueError for empty index")
           except ValueError as e:
               print(f"✓ Correctly caught empty index error: {str(e)}")

           # Test empty chunk list
           try:
               empty_retriever.index_chunks([])
               print("✘ Should have raised ValueError for empty chunk list")
           except ValueError as e:
               print(f"✓ Correctly caught empty chunk list error: {str(e)}")

           print("\n✓ All tests passed successfully!")

       except Exception as e:
           print(f"\n✘ Test failed: {str(e)}")
           raise e

   run_tests()