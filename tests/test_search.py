# -*- coding: utf-8 -*-
# test_search.py
# Description: Search functionality test script with database management.

import os
import sys
from typing import List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from colorama import init, Fore, Style
from contextual_retrieval.retriever import ContextualRetrieval
from contextual_retrieval.config import get_config

_get_config = get_config()

init()

def print_boxed(text: str, color: str = Fore.WHITE):
   """Print text in a box"""
   width = len(text) + 4
   print(f"{color}{'='*width}")
   print(f"║ {text} ║")
   print(f"{'='*width}{Style.RESET_ALL}")


def print_menu(title: str, options: List[str], show_current_query: bool = False, current_query: str = None):
    """Print menu with options"""
    print_boxed(title, Fore.GREEN)

    for i, option in enumerate(options, 1):
        # Skip showing current query if not set
        if option == "Search with Current Query":
            if current_query:
                print(f"{i}. {option}")
        else:
            print(f"{i}. {option}")
    print(f"0. Exit")


def print_result(idx: int, result: Dict, mode: str):
    """Print single search result in a box"""
    print(f"\n{Fore.YELLOW}Result #{idx}:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")

    if mode == 'rerank':
        print(f"  Relevance Score: {result['relevance_score']:.4f}")
        print(f"  Original Score: {result['original_score']:.4f}")
    else:
        print(f"  Score: {result.get('score', 0.0):.4f}")

    print(f"{Fore.CYAN}{'-' * 80}{Style.RESET_ALL}")
    print(f"  {result['content'][:200]}...")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")


def generate_test_collection(collection_name: str):
    print(f"\n{Fore.GREEN}=== Generating Test Collection ==={Style.RESET_ALL}\n")
    
    retriever = ContextualRetrieval(
        collection_name=collection_name,
        mode='contextual_embedding'
    )
    print(f"✓ System initialized for data loading")

    pdf_path = os.path.join("tests", "test_data", "test_doc.pdf")
    print("\nProcessing PDF file...")
    chunk_ids = retriever.add_document_from_pdf(pdf_path)
    print(f"✓ PDF processed successfully")
    print(f"  Generated {len(chunk_ids)} chunks")


def test_search(collection_name: str, query: str, top_k: int = 3, mode: str = 'contextual_embedding'):
    mode_map = {'1': 'contextual_embedding', '2': 'contextual_bm25', '3': 'rerank'}
    retriever = ContextualRetrieval(collection_name, mode=mode_map.get(mode, mode))

    print_boxed(f"Query: {query}", Fore.CYAN)
    results = retriever.search(query, top_k=top_k)

    print_boxed(f"Search Results ({len(results)} items)", Fore.GREEN)
    for i, result in enumerate(results, 1):
        print_result(i, result, mode)

    print_boxed("Generated Answer", Fore.MAGENTA)
    answer = retriever.generate_answer(query, results)
    print(f"{answer}\n")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")


def run_search_tests(collection_name: str):
    current_query = None
    current_top_k = None

    while True:
        # Status display
        if current_query:
            print(f"\n{Fore.CYAN}Current Query: {Style.RESET_ALL}{current_query}")
            print(f"{Fore.CYAN}Results Count: {Style.RESET_ALL}{current_top_k}")

        # Set menu options
        options = ["New Query"]
        if current_query:  # Add option to search with current query if available
            options.append("Search with Current Query")
        options.append("Back to Main Menu")

        print_boxed("Input Options", Fore.GREEN)
        # Print menu
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")
        print("0. Exit")

        choice = input(f"\n{Fore.YELLOW}Select option: {Style.RESET_ALL}")

        if choice == '0':  # Exit
            return 'exit'

        # Process user choice
        try:
            choice_num = int(choice)
            if choice_num < 0 or choice_num > len(options):
                print(f"{Fore.RED}Invalid option. Please try again.{Style.RESET_ALL}")
                continue

            if choice_num == 1:  # New Query
                current_query = input(f"\n{Fore.YELLOW}Enter your query: {Style.RESET_ALL}")
                current_top_k = int(input(f"{Fore.YELLOW}Enter number of results (default: {_get_config['search']['default_top_k']}): {Style.RESET_ALL}") or _get_config["search"]["default_top_k"])

                while True:
                    print_menu("Search Mode", [
                        "Contextual Embedding",
                        "Contextual BM25",
                        "Rerank",
                        "Back"
                    ])

                    mode_choice = input(f"\n{Fore.YELLOW}Select mode: {Style.RESET_ALL}")

                    if mode_choice == '0':  # Exit
                        return 'exit'
                    elif mode_choice == '4':  # Back
                        break
                    elif mode_choice in ['1', '2', '3']:
                        test_search(collection_name, current_query, current_top_k, mode_choice)
                        input(f"\n{Fore.YELLOW}Press Enter to continue...{Style.RESET_ALL}")
                        break

            elif choice_num == 2 and current_query:  # Search with Current Query
                while True:
                    print_menu("Search Mode", [
                        "Contextual Embedding",
                        "Contextual BM25",
                        "Rerank",
                        "Back"
                    ])

                    mode_choice = input(f"\n{Fore.YELLOW}Select mode: {Style.RESET_ALL}")

                    if mode_choice == '0':  # Exit
                        return 'exit'
                    elif mode_choice == '4':  # Back
                        break
                    elif mode_choice in ['1', '2', '3']:
                        test_search(collection_name, current_query, current_top_k, mode_choice)
                        input(f"\n{Fore.YELLOW}Press Enter to continue...{Style.RESET_ALL}")
                        break

            elif choice_num == len(options):  # Back to Main Menu
                return

        except ValueError:
            print(f"{Fore.RED}Invalid input. Please enter a number.{Style.RESET_ALL}")
            continue

def cleanup_collection(collection_name: str):
    print(f"\n{Fore.GREEN}=== Cleaning Up Collection ==={Style.RESET_ALL}")
    try:
        retriever = ContextualRetrieval(collection_name=collection_name)
        retriever.vector_store.client.delete_collection(collection_name)
        print(f"✓ Collection '{collection_name}' deleted")
    except Exception as e:
        print(f"✘ Error deleting collection: {str(e)}")


def main():
    COLLECTION_NAME = "bedrock_contextual_retrieval"

    while True:
        print_menu("Contextual Retrieval Test Menu", [
            "Generate Test Collection",
            "Run Search Tests",
            "Clean Up Collection"
        ])

        choice = input(f"\n{Fore.YELLOW}Select option: {Style.RESET_ALL}")

        try:
            if choice == '0':  # Exit
                print(f"\n{Fore.GREEN}✓ Program terminated{Style.RESET_ALL}")
                break

            elif choice == "1":
                try:
                    cleanup_collection(COLLECTION_NAME)
                    print(f"\n✓ Existing collection cleaned up")
                except:
                    pass
                generate_test_collection(COLLECTION_NAME)
                input(f"\n{Fore.YELLOW}Press Enter to continue...{Style.RESET_ALL}")

            elif choice == "2":
                result = run_search_tests(COLLECTION_NAME)
                if result == 'exit':
                    print(f"\n{Fore.GREEN}✓ Program terminated{Style.RESET_ALL}")
                    break

            elif choice == "3":
                if input(f"\n{Fore.YELLOW}Are you sure? (y/n): {Style.RESET_ALL}").lower() == 'y':
                    cleanup_collection(COLLECTION_NAME)
                    input(f"\n{Fore.YELLOW}Press Enter to continue...{Style.RESET_ALL}")

        except Exception as e:
            print(f"\n{Fore.RED}✘ Error: {str(e)}{Style.RESET_ALL}")
            input(f"\n{Fore.YELLOW}Press Enter to continue...{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
