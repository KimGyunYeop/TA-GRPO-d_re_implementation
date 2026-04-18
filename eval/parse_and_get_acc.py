import json
import re
import os
import glob
import multiprocessing
import tiktoken
from collections import defaultdict
from parser_helper import is_equiv, remove_boxed, last_boxed_only_string


def count_effective_tokens(text):
    if not text:
        return 0
    text = text.replace("<|endoftext|>", "")
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return len(tokens)


def parse_gsm_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        ground_truth = item.get("ground_truth")
        raw_generation = item.get("generations", "")
        question = item.get("question", "")

        # Count effective tokens
        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens

        parsed_answer = None

        boxed_matches = re.findall(r"\\boxed{(.*?)}", raw_generation)
        if boxed_matches:
            for boxed_content in boxed_matches:
                boxed_content = boxed_content.strip()
                if boxed_content and boxed_content != "..." and not re.match(r"^\.+$", boxed_content):
                    try:
                        parsed_answer = float(boxed_content)
                        break
                    except ValueError:
                        numbers = re.findall(r"-?\d+\.?\d*", boxed_content)
                        if numbers:
                            try:
                                parsed_answer = float(numbers[0])
                                break
                            except ValueError:
                                pass

        if parsed_answer is None:
            answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
            if answer_match:
                answer_text = answer_match.group(1).strip()
                if answer_text:
                    try:
                        parsed_answer = float(answer_text)
                    except ValueError:
                        numbers = re.findall(r"-?\d+\.?\d*", answer_text)
                        if numbers:
                            try:
                                parsed_answer = float(numbers[-1])
                            except ValueError:
                                pass

        is_correct = parsed_answer is not None and parsed_answer == ground_truth
        if is_correct:
            total_correct += 1

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": parsed_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        # total_effective_tokens,
        data["avg_diffusion_steps"]
    )


def parse_math_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        raw_generation = item.get("generations", "")

        # Count effective tokens
        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens

        parsed_answer = None

        try:
            parsed_answer = remove_boxed(last_boxed_only_string(raw_generation))
        except:
            parsed_answer = None

        if not parsed_answer:
            answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
            if answer_match:
                parsed_answer = answer_match.group(1).strip()
        is_correct = False
        if parsed_answer is not None:
            is_correct = is_equiv(parsed_answer, ground_truth)

        if is_correct:
            total_correct += 1

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": parsed_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        # total_effective_tokens,
        data["avg_diffusion_steps"]
    )


def parse_countdown_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0

    processed_items = []

    def validate_equation(equation_str, available_numbers):
        """Validate that equation only uses available numbers and each number once."""
        try:
            numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
            available_numbers = sorted(available_numbers)
            numbers_in_eq = sorted(numbers_in_eq)
            return numbers_in_eq == available_numbers
        except:
            return False

    def evaluate_equation(equation_str):
        """Safely evaluate the arithmetic equation."""
        try:
            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation_str):
                raise ValueError("Invalid characters in equation.")
            result = eval(equation_str.strip(), {"__builtins__": None}, {})
            return result
        except Exception:
            return float("Inf")

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", [])
        generated_text = item.get("generations", "")

        # Count effective tokens
        effective_tokens = count_effective_tokens(generated_text)
        total_effective_tokens += effective_tokens

        # Extract available numbers and target from ground_truth
        numbers = []
        target = None

        if isinstance(ground_truth, list) and len(ground_truth) == 2:
            numbers = ground_truth[0]
            target = ground_truth[1]
        else:
            # Fallback to parsing from question if ground_truth is not in expected format
            numbers_match = re.search(r"Numbers: \[([\d, ]+)\]", question, re.IGNORECASE)
            if numbers_match:
                numbers_str = numbers_match.group(1)
                numbers = [int(num.strip()) for num in numbers_str.split(",")]

            target_match = re.search(r"Target: (\d+)", question, re.IGNORECASE)
            if target_match:
                target = int(target_match.group(1))
        # print(generated_text)
        equation = ""
        # try:
        #     equation = remove_boxed(last_boxed_only_string(generated_text))
        # except:
        # Try to extract from answer tags
        answer_match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
        if answer_match:
            equation = answer_match.group(1).strip()
        else:
            equation = generated_text
                # print(f"Extracted using answer tags or full text: {equation}")
        # Replace LaTeX operators with Python operators
        equation = equation.replace(r"\div", "/").replace(r"\times", "*").replace(r"\cdot", "*")

        # Check for equation with equals sign and extract only the expression part
        equation_match = re.search(r"([0-9+\-*/() ]+)=[0-9. ]+", equation)
        if equation_match:
            equation = equation_match.group(1).strip()

        is_correct = False
        result = None

        # Validate and evaluate the equation
        is_valid = validate_equation(equation, numbers)
        if is_valid:
            result = evaluate_equation(equation)
            if target is not None and abs(result - target) < 1e-5:
                is_correct = True
                total_correct += 1

        processed_items.append(
            {
                "question": question,
                "extracted_answer": equation,
                "evaluation_result": result,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        # total_effective_tokens,
        data["avg_diffusion_steps"]
    )


def parse_sudoku_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct_cells = total_empty_cells = total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        raw_generation = item.get("generations", "")

        # Count effective tokens
        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens

        # Extract puzzle
        puzzle_str = ""
        if len(question) >= 16 and all(c.isdigit() or c == "0" for c in question[:16]):
            puzzle_str = question[:16]
        else:
            match = re.search(r"Sudoku puzzle: ([0-9]{16})", question)
            if match:
                puzzle_str = match.group(1)
        assert len(puzzle_str) == 16, f"Invalid puzzle string: {puzzle_str}"

        empty_indices = [i for i in range(16) if puzzle_str[i] == "0"]
        empty_cells = len(empty_indices)

        # Extract solution using regex patterns
        solution_str = ""
        patterns = [
            r"<answer>.*?```\s*([\d\s]+)```",
            r"<answer>(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|</answer>)",
            r"</answer>\s*(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|$)",
            r".*?(\d{16})\s*</answer>",
            r"\b(\d{16})\b",
        ]

        for pattern in patterns:
            if solution_str:
                break
            match = re.search(pattern, raw_generation, re.DOTALL)
            if match and match.group(1).strip():
                solution_str = match.group(1).strip()

        solution_str = re.sub(r"\s", "", solution_str)

        # Handle solution length
        if not solution_str:
            correct_cells = 0
        else:
            if len(solution_str) < 16:
                solution_str = solution_str + "0" * (16 - len(solution_str))
            elif len(solution_str) > 16:
                solution_str = solution_str[:16]
            correct_cells = sum(1 for i in empty_indices if solution_str[i] == ground_truth[i])

        accuracy = correct_cells / empty_cells if empty_cells > 0 else 0.0
        total_correct_cells += correct_cells
        total_empty_cells += empty_cells

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": solution_str,
                "ground_truth": ground_truth,
                "empty_cells": empty_cells,
                "correct_cells": correct_cells,
                "accuracy": accuracy,
                "effective_tokens": effective_tokens,
            }
        )
    return (
        total_correct_cells,
        total_empty_cells,
        processed_items,
        # total_effective_tokens * 8,
        data["avg_diffusion_steps"]
    )

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "TA-GRPO-d")))
from reward_func import run_test, split_test_function, is_safe_code

CODE_PATTERN = re.compile(r"```(?:\w+)?\n(.*?)\n```", re.DOTALL)


def extract_code_from_string(s):
    answer_match = re.search(r"<answer>(.*?)</answer>", s, re.DOTALL)
    if answer_match:
        s = answer_match.group(1).strip()
    code_blocks = CODE_PATTERN.findall(s)
    return code_blocks[-1].strip() if code_blocks else ""

def soft_extract_code_from_string(s):
    s = s.strip()
    s = s.split("<answer>")[-1]
    s = s.split("```python3")[-1]
    s = s.split("```python")[-1]
    s = s.split("</answer>")[0]
    s = s.split("```")[0]
    
    return s.strip()
    

def parse_code_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data
        

    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        test_code = item.get("ground_truth", "")
        raw_generation = item.get("generations", "")
        
        # Count effective tokens
        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens
        
        parsed_generation = extract_code_from_string(raw_generation)
        if parsed_generation is None or len(parsed_generation) == 0:
            parsed_generation = soft_extract_code_from_string(raw_generation)

        succ = False
        output = ""
        if parsed_generation:
            import_match = re.search(r"from solution import (\w+)", test_code)
            assert_match = None
            if not import_match:
                assert_match = re.search(r"assert\s+(\w+)\s*\(", test_code)
                imported_func = assert_match.group(1) if assert_match else None
            else:
                imported_func = import_match.group(1)

            solution_match = re.search(r"def\s+(\w+)\s*\(", parsed_generation)
            if solution_match is None:
                output = "No function definition found in generated code."
            else:
                defined_func = solution_match.group(1)
                solution = parsed_generation
                if imported_func is not None and defined_func != imported_func:
                    solution = re.sub(rf"\bdef {defined_func}\b", f"def {imported_func}", solution)

                if not is_safe_code(solution):
                    output = "Potentially unsafe code."
                else:
                    tests = test_code
                    test_funcs = re.findall(r"def (\w+)\(\):", tests)

                    # For eval_tmp code datasets, ground_truth is usually already a pytest-style test file.
                    # Strip solution imports and prepend the generated solution directly.
                    if "from solution import" in tests:
                        tests = re.sub(r"from solution import \*\n?", "", tests)
                        tests = re.sub(r"from solution import \w+\n?", "", tests)
                        executable = solution + "\n\n" + tests
                    elif "check(candidate)" in tests:
                        executable = solution + "\n\n" + tests + "\n\n" + f"check({defined_func})"
                    elif assert_match:
                        executable = solution + "\n\n" + tests
                    else:
                        executable = solution + "\n\n" + tests

                    if len(test_funcs) <= 1 and "check(candidate)" not in tests:
                        split_tests = split_test_function(tests)
                        if split_tests.strip():
                            executable = solution + "\n\n" + split_tests
                            test_funcs = re.findall(r"def (\w+)\(\):", split_tests)

                    if "check(candidate)" in tests:
                        wrapped_name = "test_check_wrapper"
                        executable += f"\n\ndef {wrapped_name}():\n    check({defined_func})\n"
                        test_funcs = [wrapped_name]

                    if len(test_funcs) > 0:
                        manager = multiprocessing.Manager()
                        result_dict = manager.dict()
                        exec_dir = os.path.join("execution_files", f"sample_{total_processed}")
                        jobs = []
                        for rank, fn in enumerate(test_funcs):
                            p = multiprocessing.Process(
                                target=run_test, args=(fn, executable, result_dict, exec_dir, rank)
                            )
                            p.start()
                            jobs.append(p)
                        for p in jobs:
                            p.join()

                        passed = sum(result_dict.get(fn, False) for fn in test_funcs)
                        succ = passed == len(test_funcs)
                        output = f"Passed {passed}/{len(test_funcs)} tests"
                    else:
                        output = "No executable test functions found."

        if succ:
            total_correct += 1

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": parsed_generation,
                "ground_truth": test_code,
                "is_correct": succ,
                "effective_tokens": effective_tokens,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        # total_effective_tokens,
        data["avg_diffusion_steps"]
    )


def extract_setup_name(filename):
    """Extract the setup name from the filename."""
    match = re.match(r"(.+)_\d+_generations\.json$", filename)
    if match:
        return match.group(1)
    return None


def aggregate_results(directory=".", compute_code_accuracy=False):
    """Aggregate results from all JSON files and save detailed results."""
    # Find all JSON files matching the pattern
    json_files = glob.glob(os.path.join(directory, "*_generations.json"))
    print(f"Found {len(json_files)} JSON files in {directory}.")

    # Dictionary to store aggregated results by setup
    setups = defaultdict(
        lambda: {
            "correct": 0,
            "processed": 0,
            "accuracy": 0.0,
            "questions": [],
            "total_effective_tokens": 0,
        }
    )
    os.makedirs("detailed_results", exist_ok=True)
    error_msgs = []
    for json_file in json_files:
        filename = os.path.basename(json_file)
        # setup_name = extract_setup_name(filename)
        setup_name = filename.replace("_generations.json", "")
        try:
            if setup_name:
                # print(f"Processing {filename}...")
                if "gsm" in setup_name:
                    (
                        correct,
                        processed,
                        detailed_results,
                        total_effective_tokens,
                    ) = parse_gsm_answers(json_path=json_file)
                elif "math" in setup_name:
                    (
                        correct,
                        processed,
                        detailed_results,
                        total_effective_tokens,
                    ) = parse_math_answers(json_path=json_file)
                elif "countdown" in setup_name:
                    (
                        correct,
                        processed,
                        detailed_results,
                        total_effective_tokens,
                    ) = parse_countdown_answers(json_path=json_file)
                elif "sudoku" in setup_name:
                    (
                        correct,
                        processed,
                        detailed_results,
                        total_effective_tokens,
                    ) = parse_sudoku_answers(json_path=json_file)
                elif "humaneval" in setup_name or "mbpp" in setup_name:
                    if compute_code_accuracy:
                        (
                            correct,
                            processed,
                            detailed_results,
                            total_effective_tokens,
                        ) = parse_code_answers(json_path=json_file)
                    else:
                        continue

            setups[setup_name]["correct"] += correct
            setups[setup_name]["processed"] += processed
            # setups[setup_name]["total_effective_tokens"] += total_effective_tokens
            setups[setup_name]["avg_diffusion_steps"] = total_effective_tokens
            setups[setup_name]["questions"].extend(detailed_results)
            with open(os.path.join("detailed_results", f"{setup_name}_detailed.json"), "w") as f:
                json.dump(detailed_results, f, indent=2)
        except Exception as e:
            error_msgs.append(f"Error processing {filename}: {e}")
            print(f"Error processing {filename}: {e}")
    print("\n".join(error_msgs))

    # Calculate final accuracy and save results
    for setup, results in sorted(setups.items()):
        results["accuracy"] = (
            results["correct"] / results["processed"] * 100 if results["processed"] > 0 else 0
        )
        # results["avg_effective_tokens"] = (
        #     results["total_effective_tokens"] / results["processed"] if len(results["questions"]) > 0 else 0
        # )
    # Header
    header_format = "{:<40} {:>12} {:>25}"
    print(header_format.format("Setup (task_model_seqlen_diffusteps)", "Accuracy", "Avg diffusion_steps"))
    print("-" * 80)

    # Data rows
    row_format = "{:<40} {:>11.2f}% {:>25.2f}"
    for setup, results in sorted(setups.items()):
        print(row_format.format(setup, results["accuracy"], results["avg_diffusion_steps"]))

    print("=" * 80)


if __name__ == "__main__":
    aggregate_results(directory="results", compute_code_accuracy=True)
