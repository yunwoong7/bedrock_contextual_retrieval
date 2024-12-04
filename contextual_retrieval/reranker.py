# reranker.py
# Description: Reranker for improving search results using AWS Bedrock Rerank model

import json
from typing import List, Dict, Any
import boto3
from contextual_retrieval.config import get_config

_config = get_config()

class Reranker:
   def __init__(self):
       """Initialize Bedrock client for reranking"""
       self.client = boto3.client(
           'bedrock-runtime',
           region_name=_config["bedrock"]["region"]
       )
       self.model_id = _config["bedrock"]["rerank"]["model_id"]

   def rerank(self,
             query: str,
             results: List[Dict[str, Any]],
             top_k: int = None) -> List[Dict[str, Any]]:
       """
       Rerank search results using Amazon Rerank model

       Parameters:
       - query: Original search query
       - results: List of search results (each with 'content' and 'score' keys)
       - top_k: Number of results to return (defaults to all)

       Returns:
       - List of reranked results with updated scores
       """
       try:
           # Prepare input for reranking
           documents = [result['content'] for result in results]
           request_body = {
               "query": query,
               "documents": documents,
               "top_n": top_k if top_k else len(documents)
           }

           # Call rerank model
           response = self.client.invoke_model(
               modelId=self.model_id,
               body=json.dumps(request_body),
               contentType="application/json",
               accept="*/*"
           )

           response_body = json.loads(response['body'].read())

           # Process results
           reranked_results = []
           for rank_result in response_body['results']:
               index = rank_result['index']
               original_result = results[index].copy()
               original_result['original_score'] = original_result['score']
               original_result['relevance_score'] = rank_result['relevance_score']
               original_result['score'] = rank_result['relevance_score']  # 수정된 부분
               reranked_results.append(original_result)

           return reranked_results

       except Exception as e:
           print(f"Error in reranking: {str(e)}")
           return results


if __name__ == "__main__":
   def run_tests():
       print("\n=== Running Reranker Tests ===\n")

       try:
           # Initialize reranker
           reranker = Reranker()
           print("✓ Reranker initialization successful")

           # Test search results
           test_query = "인공지능의 기초 개념"
           test_results = [
               {
                   'content': "인공지능은 인간의 학습능력과 추론능력을 컴퓨터로 구현한 기술입니다.",
                   'score': 0.85,
                   'metadata': {'category': 'AI_basics'}
               },
               {
                   'content': "머신러닝은 데이터를 기반으로 패턴을 학습하고 예측하는 기술입니다.",
                   'score': 0.75,
                   'metadata': {'category': 'machine_learning'}
               },
               {
                   'content': "딥러닝은 인공신경망을 기반으로 하는 머신러닝의 한 분야입니다.",
                   'score': 0.65,
                   'metadata': {'category': 'deep_learning'}
               }
           ]

           print("\n원본 검색 결과:")
           for i, result in enumerate(test_results, 1):
               print(f"\n  Result #{i}:")
               print(f"    Score: {result['score']:.4f}")
               print(f"    Content: {result['content']}")

           # Rerank results
           print("\n재순위화 수행중...")
           reranked_results = reranker.rerank(test_query, test_results, top_k=2)

           print("\n✓ Reranking successful")
           print(f"  Query: {test_query}")
           print(f"  Top {len(reranked_results)} results:")

           for i, result in enumerate(reranked_results, 1):
               print(f"\n  Result #{i}:")
               print(f"    Relevance Score: {result['relevance_score']:.4f}")
               print(f"    Original Score: {result['original_score']:.4f}")
               print(f"    Content: {result['content']}")

           print("\n✓ All tests passed successfully!")

       except Exception as e:
           print(f"\n✘ Test failed: {str(e)}")
           raise e

   run_tests()