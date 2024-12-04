<h2 align="center">
Bedrock Contextual Retrieval
</h2>

<div align="center">
  <img src="https://img.shields.io/badge/python-v3.12.7-blue.svg"/>
  <img src="https://img.shields.io/badge/boto3-v1.35.73-blue.svg"/>
  <img src="https://img.shields.io/badge/rank_bm25-v0.2.2-blue.svg"/>
  <img src="https://img.shields.io/badge/chromadb-v0.5.21-blue.svg"/>
</div>

AWS Bedrock 기반의 문서 검색 및 질의응답 시스템으로, Anthropic의 Contextual Retrieval 논문을 기반으로 구현되었습니다.

## Contextual Retrieval 개요

Anthropic이 2024년 9월에 발표한 Contextual Retrieval은 기존 RAG(Retrieval-Augmented Generation) 시스템의 검색 성능을 크게 향상시키는 기술입니다.

### 주요 기술 구성
1. **Contextual Embeddings**
   - 각 청크에 50-100 토큰의 맥락 정보 추가
   - 더 정확한 의미 파악 및 검색 가능

2. **Contextual BM25**
   - 맥락이 추가된 청크 기반 키워드 검색
   - 의미적 검색과 키워드 검색의 장점 결합

3. **Reranking**
   - 검색 결과 재순위화를 통한 정확도 향상

## 프로젝트 구조
```
bedrock_contextual_retrieval/
├── contextual_retrieval/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── default_config.yaml
│   │   └── schema.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── document_utils.py
│   ├── embedding_models.py
│   ├── vector_client.py 
│   ├── bm25.py
│   ├── reranker.py
│   └── retriever.py
├── tests/
│   └── test_search.py
└── asset/
    └── doc/
        └── test_doc.pdf
```

## 주요 컴포넌트

- **embedding_models.py**: AWS Bedrock Titan 임베딩 모델을 사용한 텍스트 임베딩
- **vector_client.py**: ChromaDB 기반 벡터 저장소 관리
- **bm25.py**: BM25 알고리즘 기반 텍스트 검색
- **reranker.py**: AWS Bedrock Rerank 모델을 사용한 검색 결과 재순위화
- **retriever.py**: 전체 검색 시스템 통합 및 조정

## 검색 모드

1. **Contextual Embedding**: 컨텍스트 기반 벡터 검색
   - 청크별 맥락 정보 추가
   - 의미 기반 유사도 검색

2. **Contextual BM25**: 벡터 검색 + BM25 하이브리드
   - 벡터 검색과 키워드 검색 결합
   - 더 정확한 검색 결과 제공

3. **Rerank**: 하이브리드 검색 결과 재순위화
   - 검색 결과의 정확도 향상
   - 맥락 기반 순위 조정

## 설정 (config/default_config.yaml)
```yaml
bedrock:
  region: 'us-west-2'
  llm:
    model_id: 'anthropic.claude-3-5-haiku-20241022-v1:0'
  embedding:
    model_id: 'amazon.titan-embed-text-v2:0'
  rerank:
    model_id: 'amazon.rerank-v1:0'

chromadb:
  persist_directory: "./chroma_db"
  collection_name: "contextual_retrieval"

document:
  chunk_size: 512
  chunk_overlap: 50
```

## 성능 향상
Anthropic의 연구 결과에 따르면:
- Contextual Embedding: 검색 실패율 35% 감소
- Contextual Embedding + BM25: 검색 실패율 49% 감소
- Reranking 추가: 검색 실패율 67% 감소

## 테스트 실행
```bash
python tests/test_search.py
```

### 테스트 메뉴:
1. Generate Test Collection
2. Run Search Tests
   - Contextual Embedding
   - Contextual BM25
   - Rerank
3. Clean Up Collection
4. Exit

### 데이터베이스 생성

먼저 테스트용 데이터베이스를 생성합니다. 기존에 동일한 이름의 컬렉션이 있는지 확인하고

- 있다면: 기존 컬렉션을 자동으로 삭제 후 새로 생성
- 없다면: 바로 새 컬렉션 생성 

아래는 기존 컬렉션이 있는 경우의 실행 화면입니다.

<div align="center">
<img src="https://github.com/user-attachments/assets/8e431ddc-3174-49b4-aba9-6b3894a3cdef" width="50%">
</div>

### 검색 모드별 테스트 

"Run Search Tests"를 선택하여 검색 모드별 테스트를 진행합니다. 먼저 "New Query"로 질문을 입력한 후 각 모드를 선택하여 결과를 비교할 수 있습니다. ***테스트에 사용된 문서는 "2024년 「일자리 채움 청년지원금」 (빈일자리 청년취업지원금) 사업운영 지침(안)"입니다. 이 문서를 선택한 이유는 한국어 공문서, 특히 정부 지침안의 특성인 복잡한 용어와 구조를 가진 문서에서 각 검색 모드의 성능을 비교하기 위함입니다.*** 

<div align="center">
<img src="https://github.com/user-attachments/assets/0e5417c1-196c-4314-9095-3df848b6ea37" width="50%">
</div>

### 각 모드별 테스트 결과 

#### Contextual Embedding 모드

일반적으로 Contextual Embedding 검색도 좋은 성능을 보여주지만, 이 경우에는 "제공된 검색 결과에서 일자리 채움 청년지원금의 중복 지원에 대해 명확하게 언급된 내용은 없습니다."라는 부정확한 답변을 제공했습니다.

<div align="center">
<img src="https://github.com/user-attachments/assets/a95c003c-ce7f-4d88-b8b1-a71c766041fa" width="70%">
</div>

#### Contextual Embedding + BM25 모드

BM25를 추가하자 중복 지원 관련 내용을 정확하게 찾아내어 올바른 답변을 제공했습니다.

<div align="center">
<img src="https://github.com/user-attachments/assets/22e87e99-b984-45c4-93ab-d2678f0914dd" width="70%">
</div>

#### Contextual Embedding + BM25 + Rerank 모드

Rerank 모드에서도 중복 지원 관련 정보를 정확하게 찾아 답변했습니다.

<div align="center">
<img src="https://github.com/user-attachments/assets/866f19cc-1067-4807-816b-6ba45a884068" width="70%">
</div>

테스트 결과, 단순 Contextual Embedding 검색에서는 놓친 정보를 BM25와 Rerank를 추가했을 때 정확하게 찾아낼 수 있었습니다. 특히 복잡한 공문서 내에서 예외 사항과 같은 특정 정보를 찾는 데 있어 하이브리드 검색과 재순위화가 효과적임을 확인할 수 있었습니다.

---

## 참고 문헌
- [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) - Anthropic, 2024# bedrock_contextual_retrieval
