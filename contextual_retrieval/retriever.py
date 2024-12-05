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

       # bedrock client
       self.client = boto3.client(
           'bedrock-runtime',
           region_name=_config["bedrock"]["region"]
       )

   def add_document_from_pdf(self, pdf_path: str) -> List[str]:
       """
       Index a PDF document by chunking and adding to the vector store.
       Returns a list of generated chunk IDs.

       Parameters:
       - pdf_path (str): Path to the PDF file
       """
       if not os.path.exists(pdf_path):
           raise FileNotFoundError(f"PDF file not found: {pdf_path}")

       # Read PDF text
       text = read_pdf(pdf_path)

       # Chunk the document
       chunks = chunk_document(text)

       # Index the chunks
       chunk_ids = self.add_documents(chunks, parent_doc=pdf_path)

       return chunk_ids

   def add_documents(self,
                     documents: List[str],
                     queries: Optional[List[str]] = None,
                     metadata: Optional[List[Dict[str, Any]]] = None,
                     parent_doc: Optional[str] = None) -> List[str]:
       """
       Add documents with generated context and embeddings.

       Parameters:
       - documents (List[str]): List of documents to add
       - queries (List[str], optional): Optional corresponding queries
       - metadata (List[Dict], optional): Additional metadata for each document
       - parent_doc (str, optional): Parent document path/identifier

       Returns:
       - List[str]: List of generated chunk IDs
       """
       if queries and len(queries) != len(documents):
           raise ValueError("Number of queries must match number of documents")

       # Combine documents into a single text for context generation
       full_doc = documents[0] if len(documents) == 1 else "\n\n".join(documents)

       # Chunk the document
       chunks = chunk_document(full_doc)

       # Generate contexts and embeddings
       contexts = []
       embeddings = []

       for chunk in chunks:
           # Generate context
           context = self.context_generator.generate_context(full_doc, chunk)
           contexts.append(context)

           # Generate embedding
           embedding = self.embedding_model.encode_single(context).tolist()
           embeddings.append(embedding)

       # Generate unique IDs
       chunk_ids = [f"chunk_{uuid.uuid4()}" for _ in chunks]

       # Prepare metadata
       base_metadata = []
       for i, chunk in enumerate(chunks):
           doc_metadata = {
               "timestamp": datetime.now(UTC).isoformat(),
               "chunk_id": chunk_ids[i],
               "original_text": chunk,
               "context": contexts[i],
               "source": parent_doc
           }
           if metadata and i < len(metadata):
               doc_metadata.update(metadata[i])
           base_metadata.append(doc_metadata)

       # Add to vector store
       self.vector_store.add_documents(
           documents=chunks,
           embeddings=embeddings,
           contexts=contexts,
           ids=chunk_ids,
           metadata=base_metadata
       )

       # Reset BM25 index if needed
       if hasattr(self, '_bm25_initialized'):
           delattr(self, '_bm25_initialized')

       return chunk_ids

   def search(self, query: str, top_k: int = None) -> List[Dict]:
       top_k = top_k or _config["search"]["default_top_k"]

       if self.mode == "contextual_embedding":
           return self._embedding_search(query, top_k)

       # Get more results for hybrid search and rerank
       initial_top_k = top_k * 3  # 더 많은 결과를 가져와서 재순위화

       if self.mode == "contextual_bm25":
           results = self._hybrid_search(query, initial_top_k)
           return results[:top_k]  # 최종 top_k만 반환

       else:  # rerank
           results = self._hybrid_search(query, initial_top_k)
           return self._rerank_results(query, results, top_k)

   def _initialize_bm25(self):
       """initialize BM25 retriever"""

       all_docs = self.vector_store.get_all_documents()

       texts = []
       for doc in all_docs:
           texts.append(doc['content'])

       if texts:
           self.bm25_retriever.index_chunks(texts)
           self._bm25_initialized = True

   def _embedding_search(self, query: str, top_k: int) -> List[Dict]:
       """Search using embeddings"""
       query_embedding = self.embedding_model.encode_single(query).tolist()
       raw_results = self.vector_store.search(query_embedding, top_k=top_k)

       # Formating results
       formatted_results = []
       for result in raw_results:
           formatted_result = {
               'content': result['content'],
               'score': result.get('score', 0.0),
               'metadata': result.get('metadata', {})
           }
           formatted_results.append(formatted_result)

       return formatted_results

   def _hybrid_search(self, query: str, top_k: int) -> List[Dict]:
       """Hybrid search using embeddings and BM25"""
       vector_results = self._embedding_search(query, top_k)

       original_docs = [doc['content'] for doc in self.vector_store.get_all_documents()]
       self.bm25_retriever.index_chunks(original_docs)
       bm25_results = self.bm25_retriever.retrieve(query, top_k)

       return self._combine_results(vector_results, bm25_results, top_k)

   def _combine_results(self, vector_results: List[Dict], bm25_results: List[Tuple[str, float]], top_k: int) -> List[
       Dict]:
       """
         Combine vector and BM25 search results.

            Parameters:
            - vector_results (List[Dict]): Results from vector search
            - bm25_results (List[Tuple[str, float]]): Results from BM25 search
            - top_k (int): Number of top results to return
            Returns:
            - List[Dict]: Combined search results
       """
       weights = _config["search"]["score_weights"]
       combined_scores = {}

       # Results from vector search
       for result in vector_results:
           combined_scores[result['content']] = {
               'content': result['content'],
               'score': result.get('score', 0.0) * weights['embedding'],
               'metadata': result.get('metadata', {}),
               'search_type': 'vector'
           }

       # Results from BM25 search
       for content, score in bm25_results:
           if content in combined_scores:
               # Combine scores if document is in both results
               combined_scores[content]['score'] += score * weights['bm25']
               combined_scores[content]['search_type'] = 'hybrid'
           else:
               # Add new document if not in vector results
               metadata = next((r['metadata'] for r in vector_results if r['content'] == content), {})
               combined_scores[content] = {
                   'content': content,
                   'score': score * weights['bm25'],
                   'metadata': metadata,
                   'search_type': 'bm25'
               }

       # Return top k results
       sorted_results = sorted(
           combined_scores.values(),
           key=lambda x: x['score'],
           reverse=True
       )[:top_k]

       return sorted_results

   def _rerank_results(self, query: str, results: List[Dict], top_k: int) -> List[Dict]:
       """Rerank search results"""
       return self.reranker.rerank(query, results, top_k=top_k)

   def get_stats(self) -> Dict:
       """Get collection statistics"""
       all_docs = self.vector_store.get_all_documents()
       unique_parents = len(set(
           doc['metadata'].get('parent_doc_id')
           for doc in all_docs
           if doc['metadata'].get('parent_doc_id')
       ))

       return {
           "mode": self.mode,
           "total_documents": unique_parents,
           "total_chunks": len(all_docs),
           "vector_store": self.vector_store.get_collection_stats()
       }

   def generate_answer(self, query: str, search_results: List[Dict]) -> str:
       """Generate answer using search results"""

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