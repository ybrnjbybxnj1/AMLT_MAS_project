import os
import sys
from pathlib import Path
import argparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from graph import build_graph, run_query

"""
Main entry point for guys who like terminal
Usage: python -m src.main "Your research query here"
   or: python src/main.py "Your research query here" (from project root)
"""
sys.path.insert(0, str(Path(__file__).parent))


def main():
    # load environment variables from project root .env
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    load_dotenv(env_path)
    BASE_URL = os.getenv("LITELLM_BASE_URL", LITELLMM BASE URL)
    API_KEY = os.getenv("LITELLM_API_KEY", LITELLMM API KEY)
    MODEL_NAME = os.getenv("MODEL_NAME", "qwen3-32b")
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Hypothesis Planner"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Research query to process"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    args = parser.parse_args()
    # initialize LLM
    print(f"Initializing LLM: {MODEL_NAME}")
    llm = ChatOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL_NAME,
        temperature=0.7,
        max_tokens=2000
    )
    # build graph
    graph = build_graph(llm)
    if args.interactive:
        # interactive mode
        print("---")
        print("Multi-Agent Research Hypothesis Planner")
        print("Type 'quit' or 'exit' to stop")
        print("---")
        while True:
            try:
                query = input("Enter your research query: ").strip()
                if query.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                if not query:
                    continue   
                run_query(graph, query)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    elif args.query:
        # single query mode
        run_query(graph, args.query)
    else:
        # no query provided - show help
        parser.print_help()
        print("Example:")
        print('  python src/main.py "What are benefits of multi-agent systems?"')
        print('  python src/main.py --interactive')
if __name__ == "__main__":
    main()
