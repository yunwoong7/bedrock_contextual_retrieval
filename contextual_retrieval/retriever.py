# retriever.py
# Description: Main retrieval system that integrates vector search, BM25, and reranking.

import uuid
import os
from datetime import datetime, UTC
from typing import List, Dict, Any, Optional, Tuple, Literal
import json
import boto3
from contextual_retrieval.embedding_models import get_embedding_model
from contextual_retrieval.context_generator import ContextGenerator
from contextual_retrieval.vector_client import VectorClient
from contextual_retrieval.bm25 import BM25Retriever
from contextual_retrieval.reranker import Reranker
from contextual_retrieval.config import get_config
from contextual_retrieval.utils import read_pdf, chunk_document

_config = get_config()


class ContextualRetrieval:
   def __init__(self,
                collection_name: str = None,
                mode: str = None):
       """
       Initialize the contextual retrieval system.
       """
       self.collection_name = collection_name or _config["chromadb"]["collection_name"]
       self.mode = mode or _config["search"]["default_mode"]

       if self.mode not in ['contextual_embedding', 'contextual_bm25', 'rerank']:
           raise ValueError("Invalid mode. Choose from 'contextual_embedding', 'contextual_bm25', or 'rerank'")

       self.embedding_model = get_embedding_model()
       self.context_generator = ContextGenerator()
       self.vector_store = VectorClient(self.collection_name)
       self.bm25_retriever = BM25Retriever()
       self.reranker = Reranker()

       self.documents = []
       self.chunks = []
       self.parent_mappings = {}  # chunk_id -> parent_doc_id mapping'

       # Bedrock 클라이언트 초기화 추가
       self.client = boto3.client(
           'bedrock-runtime',
           region_name=_config["bedrock"]["region"]
       )

   def add_document_from_pdf(self, pdf_path: str) -> List[str]:
       """
       PDF 파일을 읽어서 처리하고 인덱싱
       """
       if not os.path.exists(pdf_path):
           raise FileNotFoundError(f"PDF file not found: {pdf_path}")

       # PDF 읽기
       text = read_pdf(pdf_path)

       # 청크로 분할
       chunks = chunk_document(text)

       # 인덱싱
       return self.add_documents(chunks, parent_doc=pdf_path)

   def add_documents(self,
                     documents: List[str],
                     queries: Optional[List[str]] = None,
                     metadata: Optional[List[Dict[str, Any]]] = None,
                     parent_doc: Optional[str] = None) -> List[str]:
       """
       문서 추가 및 인덱싱
       """
       if queries and len(queries) != len(documents):
           raise ValueError("Number of queries must match number of documents")

       # 문서 청킹
       chunks = chunk_document(documents[0] if len(documents) == 1 else "\n\n".join(documents))

       # Generate contexts and embeddings
       contexts = []
       embeddings = []

       for chunk in chunks:
           # 컨텍스트 생성
           context = self.context_generator.generate_context("", chunk)
           contexts.append(context)

           # 임베딩 생성
           embedding = self.embedding_model.encode_single(context).tolist()
           embeddings.append(embedding)

       # 청크 ID 생성
       chunk_ids = [f"chunk_{uuid.uuid4()}" for _ in chunks]

       # 메타데이터 준비
       base_metadata = []
       for i, chunk in enumerate(chunks):
           doc_metadata = {
               "timestamp": datetime.now(UTC).isoformat(),
               "context": contexts[i],
               "chunk_id": chunk_ids[i],
               "source": parent_doc
           }
           if metadata and i < len(metadata):
               doc_metadata.update(metadata[i])
           base_metadata.append(doc_metadata)

       # 벡터 스토어에 추가
       self.vector_store.add_documents(
           documents=chunks,
           embeddings=embeddings,
           contexts=contexts,
           ids=chunk_ids,
           metadata=base_metadata
       )

       # BM25 초기화 상태 리셋 (다음 검색 시 재초기화)
       if hasattr(self, '_bm25_initialized'):
           delattr(self, '_bm25_initialized')

       return chunk_ids

   def search(self, query: str, top_k: int = None) -> List[Dict]:
       top_k = top_k or _config["search"]["default_top_k"]

       if self.mode == "contextual_embedding":
           return self._embedding_search(query, top_k)

       # hybrid나 rerank는 더 많은 결과를 가져옴
       initial_top_k = top_k * 3  # 더 많은 결과를 가져와서 재순위화

       if self.mode == "contextual_bm25":
           results = self._hybrid_search(query, initial_top_k)
           return results[:top_k]  # 최종 top_k만 반환

       else:  # rerank
           results = self._hybrid_search(query, initial_top_k)
           return self._rerank_results(query, results, top_k)

   def _initialize_bm25(self):
       """BM25 인덱스 초기화 - 원본 문서와 컨텍스트 모두 사용"""
       all_docs = self.vector_store.get_all_documents()

       # 원본 문서와 컨텍스트 모두 포함
       texts = []
       for doc in all_docs:
           texts.append(doc['content'])  # 원본 문서
           context = doc['metadata'].get('context')
           if context:
               texts.append(context)  # 컨텍스트

       if texts:
           self.bm25_retriever.index_chunks(texts)
           self._bm25_initialized = True

   def _embedding_search(self, query: str, top_k: int) -> List[Dict]:
       """임베딩 기반 검색"""
       query_embedding = self.embedding_model.encode_single(query).tolist()
       raw_results = self.vector_store.search(query_embedding, top_k=top_k)

       # 결과 형식 통일화
       formatted_results = []
       for result in raw_results:
           formatted_result = {
               'content': result['content'],
               'score': result.get('score', 0.0),  # score 키를 일관되게 사용
               'metadata': result.get('metadata', {})
           }
           formatted_results.append(formatted_result)

       return formatted_results

   def _hybrid_search(self, query: str, top_k: int) -> List[Dict]:
       # 더 많은 결과를 가져옴
       vector_results = self._embedding_search(query, top_k)

       original_docs = [doc['content'] for doc in self.vector_store.get_all_documents()]
       self.bm25_retriever.index_chunks(original_docs)
       bm25_results = self.bm25_retriever.retrieve(query, top_k)

       return self._combine_results(vector_results, bm25_results, top_k)

   def _combine_results(self, vector_results: List[Dict], bm25_results: List[Tuple[str, float]], top_k: int) -> List[
       Dict]:
       """
       임베딩과 BM25 결과 결합

       Parameters:
       - vector_results: 벡터 검색 결과 리스트 (딕셔너리 형태)
       - bm25_results: BM25 검색 결과 리스트 (튜플 형태: (content, score))
       - top_k: 반환할 결과 수
       """
       weights = _config["search"]["score_weights"]
       combined_scores = {}

       # 벡터 검색 결과 처리
       for result in vector_results:
           combined_scores[result['content']] = {
               'content': result['content'],
               'score': result.get('score', 0.0) * weights['embedding'],
               'metadata': result.get('metadata', {}),
               'search_type': 'vector'
           }

       # BM25 결과 처리
       for content, score in bm25_results:
           if content in combined_scores:
               # 이미 있는 문서면 점수 합산
               combined_scores[content]['score'] += score * weights['bm25']
               combined_scores[content]['search_type'] = 'hybrid'
           else:
               # 새로운 문서면 추가
               metadata = next((r['metadata'] for r in vector_results if r['content'] == content), {})
               combined_scores[content] = {
                   'content': content,
                   'score': score * weights['bm25'],
                   'metadata': metadata,
                   'search_type': 'bm25'
               }

       # 정렬 및 상위 결과 반환
       sorted_results = sorted(
           combined_scores.values(),
           key=lambda x: x['score'],
           reverse=True
       )[:top_k]

       return sorted_results

   def _rerank_results(self, query: str, results: List[Dict], top_k: int) -> List[Dict]:
       """결과 재순위화"""
       return self.reranker.rerank(query, results, top_k=top_k)

   def get_stats(self) -> Dict:
       """시스템 통계 정보 반환"""
       all_docs = self.vector_store.get_all_documents()
       unique_parents = len(set(
           doc['metadata'].get('parent_doc_id')
           for doc in all_docs
           if doc['metadata'].get('parent_doc_id')
       ))

       return {
           "mode": self.mode,
           "total_documents": unique_parents,  # 고유한 부모 문서 수
           "total_chunks": len(all_docs),  # 총 청크 수
           "vector_store": self.vector_store.get_collection_stats()
       }

   def generate_answer(self, query: str, search_results: List[Dict]) -> str:
       """검색 결과를 기반으로 답변 생성"""
       # 검색 결과를 문맥으로 구성
       contexts = []
       for result in search_results:
           contexts.append(f"Content: {result['content']}")
       context_text = "\n\n".join(contexts)

       # 프롬프트 구성
       prompt = f"""다음은 질문과 관련된 검색 결과입니다. 이를 참고하여 질문에 답변해주세요.

   검색 결과:
   {context_text}

   질문: {query}

   위 검색 결과를 바탕으로 답변해주세요. 검색 결과에 없는 내용은 답변에 포함하지 마세요."""

       try:
           response = self.client.invoke_model(
               modelId=_config["bedrock"]["answer"]["model_id"],
               body=json.dumps({
                   "anthropic_version": "bedrock-2023-05-31",
                   "messages": [{
                       "role": "user",
                       "content": prompt
                   }],
                   "max_tokens": _config["bedrock"]["answer"]["max_tokens"],
                   "temperature": _config["bedrock"]["answer"]["temperature"]
               })
           )

           response_body = json.loads(response['body'].read())
           return response_body['content'][0]['text']

       except Exception as e:
           raise Exception(f"Error generating answer: {str(e)}")


if __name__ == "__main__":
    def generate_test_collection(collection_name: str):
        """테스트 컬렉션 생성 및 데이터 로드"""
        print("\n=== Generating Test Collection ===\n")

        # 기본 모드로 초기화
        retriever = ContextualRetrieval(
            collection_name=collection_name,
            mode='contextual_embedding'
        )
        print(f"✓ System initialized for data loading")

        # PDF 파일 처리
        pdf_path = "/tests/test_data/doc/test_doc.pdf"
        print("\nPDF 파일 처리 중...")
        chunk_ids = retriever.add_document_from_pdf(pdf_path)
        print(f"✓ PDF processed successfully")
        print(f"  Generated {len(chunk_ids)} chunks")


    def run_search_tests(collection_name: str):
        current_query = None
        current_top_k = None

        while True:
            if current_query:
                print(f"\n현재 질문: {current_query}")
                print(f"결과 수: {current_top_k}")

            print("\n입력 옵션:")
            print("1. 새로운 질문 입력")
            print("2. 기존 질문으로 검색 모드 선택")
            print("3. 이전 메뉴로 돌아가기")

            choice = input("\n선택 (1-3): ")

            if choice == '3':
                break

            elif choice == '1':
                current_query = input("\n질문을 입력하세요: ")
                current_top_k = int(input("반환할 결과 수를 입력하세요 (기본값: 3): ") or 3)

            if current_query:  # 질문이 있을 때만 검색 모드 선택 가능
                while True:
                    print("\n검색 모드 선택:")
                    print("1. Contextual Embedding")
                    print("2. Contextual BM25")
                    print("3. Rerank")
                    print("4. 이전으로 돌아가기")

                    mode_choice = input("\n모드 선택 (1-4): ")

                    if mode_choice == '4':
                        break

                    if mode_choice in ['1', '2', '3']:
                        mode_map = {
                            '1': 'contextual_embedding',
                            '2': 'contextual_bm25',
                            '3': 'rerank'
                        }

                        mode = mode_map[mode_choice]

                        # 검색 실행
                        retriever = ContextualRetrieval(
                            collection_name=collection_name,
                            mode=mode
                        )

                        print(f"\n검색어: {current_query}")
                        results = retriever.search(current_query, top_k=current_top_k)

                        print(f"\n검색 결과 ({len(results)} items):")
                        for i, result in enumerate(results, 1):
                            print(f"\n  Result #{i}:")
                            if mode == 'rerank':
                                print(f"    Relevance Score: {result['relevance_score']:.4f}")
                                print(f"    Original Score: {result['original_score']:.4f}")
                            else:
                                print(f"    Score: {result.get('score', 0.0):.4f}")
                            print(f"    Content: {result['content'][:200]}...")

                        print("\n검색 결과를 바탕으로 생성된 답변:")
                        try:
                            answer = retriever.generate_answer(current_query, results)
                            print(f"{answer}\n")
                            print("-" * 80)
                        except Exception as e:
                            print(f"답변 생성 실패: {str(e)}")
                    else:
                        print("잘못된 선택입니다. 1-4 중 선택해주세요.")
            elif choice == '2':
                print("\n먼저 질문을 입력해주세요.")


    def cleanup_collection(collection_name: str):
        """테스트 컬렉션 정리"""
        print("\n=== Cleaning Up Test Collection ===\n")

        try:
            retriever = ContextualRetrieval(collection_name=collection_name)
            retriever.vector_store.client.delete_collection(collection_name)
            print(f"✓ Collection '{collection_name}' deleted")
        except Exception as e:
            print(f"✘ Error deleting collection '{collection_name}': {str(e)}")


    def print_menu():
        print("\n=== Contextual Retrieval Test Menu ===")
        print("1. Generate Test Collection")
        print("2. Run Search Tests")
        print("3. Clean Up Collection")
        print("4. Exit")
        return input("\nSelect an option (1-4): ")


    def main():
        COLLECTION_NAME = "bedrock_contextual_retrieval"

        while True:
            choice = print_menu()

            try:
                if choice == "1":
                    # 기존 컬렉션이 있다면 삭제
                    try:
                        cleanup_collection(COLLECTION_NAME)
                        print(f"\n✓ Existing collection '{COLLECTION_NAME}' cleaned up")
                    except:
                        print(f"\nNote: No existing collection to clean up")

                    # 새 컬렉션 생성
                    print(f"\nGenerating collection: {COLLECTION_NAME}")
                    generate_test_collection(COLLECTION_NAME)
                    print(f"\n✓ Collection generated successfully")

                elif choice == "2":
                    print(f"\nRunning tests on collection: {COLLECTION_NAME}")
                    try:
                        run_search_tests(COLLECTION_NAME)
                    except Exception as e:
                        print(f"\n✘ Test error: {str(e)}")
                        print("Continue with next test...")

                elif choice == "3":
                    if input("\n⚠️ Are you sure you want to clean up? (y/n): ").lower() == 'y':
                        cleanup_collection(COLLECTION_NAME)
                        print(f"\n✓ Collection '{COLLECTION_NAME}' cleaned up")
                    else:
                        print("\nCleanup cancelled")

                elif choice == "4":
                    if input("\n⚠️ This will exit the program. Clean up collection? (y/n): ").lower() == 'y':
                        try:
                            cleanup_collection(COLLECTION_NAME)
                            print(f"\n✓ Collection cleaned up")
                        except:
                            pass
                    print("\n✓ Goodbye!")
                    break

                else:
                    print("\n⚠️ Invalid choice. Please select 1-4.")

            except Exception as e:
                print(f"\n✘ Error: {str(e)}")
                print("You can continue with other operations.")

    main()