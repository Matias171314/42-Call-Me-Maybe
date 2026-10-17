import argparse
import json
import sys
import time
from pathlib import Path

from src.model_engine import Model
from src.function_calling_engine import FunctionCallingEngine

from src.parsing.function_definition import (
    FunctionDefinitionError,
    get_function_definitions,
)

from src.parsing.prompt import (
    PromptError,
    get_prompts,
)


def get_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Call Me Maybe - Function Calling Engine"
    )

    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the JSON file containing function definitions.",
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the JSON file containing input prompts.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path where the generated results will be written.",
    )

    return parser.parse_args()


def main() -> None:
    """Main execution flow for the prompt processor."""
    args: argparse.Namespace = get_arguments()
    started = time.perf_counter()

    print("[INFO] Hello from 42-call-me-maybe!")
    print("[INFO] Starting Program...")

    functions_file = Path(args.functions_definition)
    input_file = Path(args.input)
    output_file = Path(args.output)

    print(f"[INFO] Functions definition: {functions_file}")
    print(f"[INFO] Input file: {input_file}")
    print(f"[INFO] Output file: {output_file}")

    try:
        functions = get_function_definitions(str(functions_file))
        prompts = get_prompts(str(input_file))
        print(
            f"[SUCCESS] Loaded {len(functions)} functions & "
            f"{len(prompts)} prompts."
        )
    except (FunctionDefinitionError, PromptError) as e:
        print(
            f"\033[0;31m[CRITICAL] Error on input files:\n{e}\033[0m",
            file=sys.stderr
        )
        sys.exit(1)

    try:
        if prompts:
            print("[INFO] Loading LLM Engine...")
            llm = Model()
            engine = FunctionCallingEngine(prompts, functions, llm)
            output = engine.process()
        else:
            output = []
    except Exception as error:
        print(f"[CRITICAL] Processing failed: {error}", file=sys.stderr)
        sys.exit(1)

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False, allow_nan=False)
        print(f"[SUCCESS] Results saved successfully in: {output_file}")
        print(f"[INFO] Total time: {time.perf_counter() - started:.2f}s")
        sys.exit(0)
    except (OSError, ValueError, UnicodeError) as e:
        print(
            f"\033[0;31m[ERROR] Could not write to output. "
            f"Details: {e}\033[0m",
            file=sys.stderr
        )
        sys.exit(1)


if __name__ == '__main__':
    main()
