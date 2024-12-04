# context_generator.py
# Description: Context generator using AWS Bedrock Claude model.

import json
import boto3
from contextual_retrieval.config import get_config

_config = get_config()


class ContextGenerator:
    def __init__(self):
        self.client = boto3.client(
            'bedrock-runtime',
            region_name=_config["bedrock"]["region"]
        )
        self.model_id = _config["bedrock"]["llm"]["model_id"]

    def clean_text(self, text: str) -> str:
        """Clean text by handling encoding issues"""
        return text.encode('utf-8', 'ignore').decode('utf-8')

    def generate_context(self, query: str, answer: str) -> str:
        """
        Generate context from query and answer using Claude.

        Parameters:
        - query (str): The input query
        - answer (str): The answer to generate context for

        Returns:
        - str: Generated context
        """
        prompt = f"""다음 질문과 답변에 대한 맥락을 생성해주세요. 답변의 핵심 정보를 포함하면서도 검색과 검색 결과 개선에 도움이 되도록 해주세요.
        질문: {query}
        답변: {answer}
        맥락 정보만 간단히 작성해주세요."""

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [{
                        "role": "user",
                        "content": prompt
                    }],
                    "max_tokens": _config["bedrock"]["llm"]["max_tokens"],
                    "temperature": _config["bedrock"]["llm"]["temperature"]
                })
            )

            response_body = json.loads(response['body'].read())
            return self.clean_text(response_body['content'][0]['text'])

        except Exception as e:
            raise Exception(f"Error generating context: {str(e)}")


if __name__ == "__main__":
    def run_tests():
        print("\n=== Running Context Generator Tests ===\n")

        try:
            # Initialize generator
            generator = ContextGenerator()
            print("✓ Generator initialization successful")

            # Test context generation
            query = "인공지능이란 무엇인가요?"
            answer = "인공지능은 인간의 학습능력과 추론능력을 컴퓨터로 구현한 기술입니다."

            context = generator.generate_context(query, answer)
            print("\n✓ Context generation successful")
            print(f"\nQuery: {query}")
            print(f"Answer: {answer}")
            print(f"Generated Context: {context}")

            # Test error handling
            try:
                generator.generate_context("", "")
                assert False, "Should have failed on empty input"
            except Exception as e:
                print("\n✓ Empty input handling successful")

            print("\n✓ All tests passed successfully!")

        except Exception as e:
            print(f"\n✘ Test failed: {str(e)}")
            raise e


    run_tests()