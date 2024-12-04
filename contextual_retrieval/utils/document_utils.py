# document_utils.py
# Description: Document processing utilities for reading and chunking documents.

import PyPDF2
import re
from typing import List


def read_pdf(file_path: str) -> str:
    """PDF 파일 읽기"""
    try:
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Error reading PDF file: {str(e)}")


def chunk_document(
        text: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separator: str = "\n"
) -> List[str]:
    """
    문서를 청크로 분할

    Args:
        text: 분할할 텍스트
        chunk_size: 청크 크기
        chunk_overlap: 청크 간 중복 크기
        separator: 분할 구분자
    """
    if not text:
        return []

    chunks = []
    paragraphs = text.split(separator)
    current_chunk = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        words = paragraph.split()
        paragraph_length = len(words)

        if current_length + paragraph_length <= chunk_size:
            current_chunk.append(paragraph)
            current_length += paragraph_length
        else:
            if current_chunk:
                chunks.append(separator.join(current_chunk))
            current_chunk = [paragraph]
            current_length = paragraph_length

    if current_chunk:
        chunks.append(separator.join(current_chunk))

    return chunks


def clean_text(text: str) -> str:
    """텍스트 정제"""
    # 여러 줄 공백 제거
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # 특수문자 처리
    text = re.sub(r'[^\w\s\n.,:;?!()\-\'\"]+', ' ', text)
    # 여러 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    return text.strip()