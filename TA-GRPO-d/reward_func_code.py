import re
from typing import Tuple, Optional
import time
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

_ERROR_MSG_PREFIX = "Failed to execute program: "
_DEFAULT_TIMEOUT_SECONDS = 20

_MAX_CHAR_DISPLAY = 2048
CODER1_EXEC = os.environ.get("CODER1_EXEC", "firejail")

import os
import subprocess

from tempfile import NamedTemporaryFile, TemporaryDirectory


# So I tried 4 approaches for code execution (after a few all-nighters...):
# 1. _remote_code_exec_ces -- Directly using https://github.com/cassanof/code_exec_server
#       - Is fast but leads to unreasonable false positives of timeouts
#       - I tried to alleviate this by (i) restarting the server frequently; (ii) bigger timeout; (iii) lower concurrency
#       - Still feels 10% false positives and bad concurrency
# 2. _remote_code_exec_kira -- Extending https://github.com/cassanof/code_exec_server to support my format and use some OS features for isolation
#       - Less unreasonable timeouts but the concurrency is very bad, stucking at create temp dirs all the time
# 3. https://e2b.dev/
#       - Concurrency is fine
#       - Does not support STDIN by default - needs some hack to support it
#       - I don't want to pay other servers when I have 192 physical CPUs...
# 4. _code_exec_firejail -- Using firejail (https://github.com/netblue30/firejail)
#       - User space isolation (some ulimit/rlimit features)
#       - Drop-in OS isolation via seccomp (blocking socket, etc.)
#       - Concurrency is the best so far
#       - This is not the safest - but docker is not safe either :L. Looks good enough for my dataset anyways.
# sudo add-apt-repository ppa:deki/firejail
# sudo apt-get update
# sudo apt-get install firejail firejail-profiles

CLI_ARG_SIZE_LIMIT = 1024 * 3

def timeout(timeout_seconds: int = 8):
    if os.name == "posix":
        import signal

        def decorator(func):

            def handler(signum, frame):
                raise TimeoutError("Operation timed out!")

            def wrapper(*args, **kwargs):
                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout_seconds)

                try:
                    return func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

            return wrapper

        return decorator
    else:
        raise NotImplementedError(f"Unsupported OS: {os.name}")


# @timeout(timeout_seconds=10)
def code_exec_firejail(code, stdin: str = None, timeout=_DEFAULT_TIMEOUT_SECONDS, pytest: str = None):
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    if "PYTHONPATH" in env:
        del env["PYTHONPATH"] # avoid importing wrong stuff

    # Build the firejail command with resource limits and cleanup options
    command = [
        "firejail",
        "--private",
        "--quiet",
        "--seccomp=socket",
        "--rlimit-nproc=32",
        "--rlimit-nofile=32",
        "--rlimit-fsize=2m",  # Limit file size
        "--rlimit-as=4096m",
        f"--timeout=00:00:{timeout}",
    ]

    if pytest:
        # solution is in {tmpdir}/solution.py
        with TemporaryDirectory() as tmpdir:
            assert stdin is None, "STDIN is not supported with pytest"
            # Write the solution to a file
            with open(os.path.join(tmpdir, "solution.py"), "w") as f:
                f.write(code)
            with open(os.path.join(tmpdir, "test_solution.py"), "w") as f:
                f.write(pytest)
            command.insert(4, f"--whitelist={tmpdir}")
            command.extend(["python3", "-m", "pytest", tmpdir])
            result = subprocess.run(
                command,
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                check=False,
            )
    else:
        if len(code) < CLI_ARG_SIZE_LIMIT:
            command.extend(["python3", "-c", code])
            result = subprocess.run(command,
                                    input=stdin.encode() if stdin else None,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    env=env,
                                    check=False)
        else:
            with NamedTemporaryFile() as tmp:
                tmp.write(code.encode())
                tmp.flush()
                command.insert(4, f"--whitelist={tmp.name}")
                command.extend(["python3", tmp.name])
                result = subprocess.run(command,
                                        input=stdin.encode() if stdin else None,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        env=env,
                                        check=False)
    
    stderr = result.stderr.decode().strip()
    stdout = result.stdout.decode()

    if result.returncode == 0:
        return True, stdout
    return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"

code_exec = code_exec_firejail

def remote_check_stdio(code, stdin, stdout):
    succ, output = code_exec(code=code, stdin=stdin)
    return succ, output, stdin, stdout


def validate_response_structure(processed_str: str) -> bool:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    return bool(pattern.match(processed_str.strip()))


# https://github.com/Unakar/Logic-RL/blob/main/verl/utils/reward_score/kk.py
def try_extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    if matches:
        final_answer = matches[-1].group(1).strip()
        return final_answer

    return solution_str


CODE_PATTERN = re.compile(r'```(?:\w+)?\n(.*?)\n```', re.DOTALL)


def extract_code_from_string(solution_str):
    solution_str = try_extract_solution(solution_str)
    code_blocks = CODE_PATTERN.findall(solution_str)
    
    # return '\n'.join(code_blocks).strip()
    return code_blocks[-1].strip() if code_blocks else "" # we will take only the last code block

def _compute_score(solution_str, ground_truth, extra_info, format_reward=0.1, answer_reward=1.):
    reward_log = []

    # ground_truth is not code, but tests
    pass_fmt = validate_response_structure(solution_str)
    solution_code = extract_code_from_string(solution_str)

    if format_reward > 0:
        if not pass_fmt or len(solution_code) == 0:  # only print full output when there is an error
            reward_log.append("\nBad format detected!")
            reward_log.append("Original Model Output:")
            reward_log.append("-" * 32)
            reward_log.append(solution_str)
            reward_log.append("-" * 32)
            return -answer_reward - format_reward, "\n".join(reward_log)

    # reward_log.append("-" * 16 + "Extracted Code to Execute" + "-" * 16)
    ground_truth = json.loads(ground_truth)

    # log code
    if "functional" in ground_truth:
        reward_log.append(solution_code + "\n" + ground_truth["functional"])
    else:
        reward_log.append(solution_code)
        
    if len(solution_code) == 0:
        reward_log.append("\nNo code extracted!")
        return format_reward, "\n".join(reward_log)

    t_start = time.time()

    if "pytest" in ground_truth or "functional" in ground_truth:
        if "functional" in ground_truth:
            succ, output = code_exec(solution_code + "\n" + ground_truth["functional"])
        else:  # pytest
            succ, output = code_exec(solution_code, pytest=ground_truth["pytest"])
        if not succ:
            reward_log.append("!" * 16 + f"⚠️ Test Execution Failed in {time.time() - t_start:.1f}s" + "!" * 16)
            reward_log.append(output[:_MAX_CHAR_DISPLAY])
            reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
            reward_log.append(extra_info["prompt"].replace("\n\n", "\n"))
            return format_reward, "\n".join(reward_log)
    elif "inputs" in ground_truth and "outputs" in ground_truth:
        stdin_list: str = ground_truth["inputs"]
        stdout_list: str = ground_truth["outputs"]

        # Add parallelism
        with ThreadPoolExecutor(max_workers=min(4, len(stdin_list))) as executor:
            futures = [
                executor.submit(remote_check_stdio, solution_code, stdin, stdout)
                for stdin, stdout in zip(stdin_list, stdout_list)
            ]
            for future in as_completed(futures):
                succ, output, stdin, stdout = future.result()
                if not succ or output.strip() != stdout.strip():
                    # output = output[:_MAX_CHAR_DISPLAY]  # truncate output to print
                    # reward_log.append("!" * 16 + f"⚠️ Test Execution Failed in {time.time() - t_start:.1f}s" + "!" * 16)
                    # reward_log.append(f"🔎Input: {repr(stdin)}")
                    reward_log.append(f"---Error---")
                    reward_log.append(f"Expected: {repr(stdout.strip())}")
                    reward_log.append(
                        f"Actual: {output if output.startswith(_ERROR_MSG_PREFIX) else repr(output.strip())}")
                    # reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
                    # reward_log.append(extra_info["prompt"].replace("\n\n", "\n"))
                    return format_reward, "\n".join(reward_log)
    else:
        raise ValueError(
            f"Current supports for ground-truth are ['functional', 'inputs/outputs'] -- No idea what's: {ground_truth = }"
        )

    reward_log.append("+" * 16 + "Test Execution Passed! (Output)" + "+" * 16)
    reward_log.append(output)
    return format_reward + answer_reward, "\n".join(reward_log)

def compute_score(solution_str, ground_truth, extra_info, solution, format_reward=0.0, answer_reward=1., debug=True):
    if isinstance(extra_info, np.ndarray):
        extra_info = extra_info.item()
    score, reward_log = _compute_score(solution_str,
                                       ground_truth,
                                       extra_info=extra_info,
                                       format_reward=format_reward,
                                       answer_reward=answer_reward)
    # we will use custom format reward
    marker = "✅" if score == (format_reward + answer_reward) else "❌"
    if debug:
        print(f"=" * 60)
        reward_log = "Reward Summary " + marker * 1 + "\nReward Log:" + reward_log + "\nSolution:" + solution + "\n" + f"Final Reward = {score} " + marker * 1
        print(reward_log + "\n")
        print(f"=" * 60)
    else:
        reward_log = f"Reward Summary {marker} /// Final Reward = {score}"
        print(reward_log)
    return score, reward_log

def kodcode_reward_func(prompts, completions, ground_truth, extra_info, solution, step=None, run_name=None, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    
    # print(f"\n\nresponses = {responses}")
    # print(f"\n\nq = {q}")
    # print(f"\n\nground_truth = {ground_truth}")
    # print(f"\n\nextra_info = {extra_info}")
    
    scores = []
    reward_logs = []
    for response, gt, ei, sol in zip(responses, ground_truth, extra_info, solution):
        score, reward_log = compute_score(response, gt, ei, sol)
        scores.append(score)
        reward_logs.append(reward_log)
        
    return scores


def strict_format_reward_func_for_kodcode(completions, **kwargs) -> list[float]:
    pattern = r"^<think>\s*.*?\s*</think>\s*<answer>\s*.*?\s*</answer>$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func_for_kodcode(completions, **kwargs) -> list[float]:
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]
