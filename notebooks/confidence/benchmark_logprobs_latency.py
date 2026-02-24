"""
Benchmark script to measure latency overhead of using logprobs with OpenAI API.

This script tests the performance difference between standard API calls and
calls that request log probabilities, helping quantify the actual overhead.

Design considerations:
- Alternates between with/without logprobs to avoid time-based confounds
- Randomizes prompts to prevent caching
- Uses smaller sample size (n=5) for efficiency
"""

import time
import statistics
from openai import OpenAI
import os
import random

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Test configuration
MODEL = "gpt-4o" #, "gpt-4o-mini"
N_ITERATIONS = 50  # Number of test iterations per condition

# Diverse prompts to avoid caching effects
PROMPT_POOL = [
    {"item": "brick", "response": "use it as a paperweight"},
    {"item": "paperclip", "response": "bend it into a sculpture"},
    {"item": "shoe", "response": "use as a flowerpot"},
    {"item": "bottle", "response": "create a musical instrument"},
    {"item": "book", "response": "use as a doorstop"},
    {"item": "rope", "response": "make a jump rope"},
    {"item": "box", "response": "wear it as a hat"},
    {"item": "pencil", "response": "use as a drumstick"},
    {"item": "spoon", "response": "use as a mirror"},
    {"item": "tire", "response": "make a swing"},
]

def create_prompt(item, response):
    """Create a prompt for creativity scoring."""
    return [
        {"role": "system", "content": "You are scoring creativity on a 1-5 scale."},
        {"role": "user", "content": f"Item: {item}\nResponse: {response}\nScore:"}
    ]

def run_api_call(messages, use_logprobs=False, top_logprobs=5):
    """Run a single API call with timing."""
    start_time = time.perf_counter()

    if use_logprobs:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=10,
            temperature=0.0,
            logprobs=True,
            top_logprobs=top_logprobs
        )
    else:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=10,
            temperature=0.0
        )

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    return latency_ms

def benchmark_alternating(n_iterations=N_ITERATIONS, top_logprobs=5):
    """
    Benchmark with alternating conditions to control for time-based effects.
    Randomizes prompts to prevent caching.
    """
    without_logprobs = []
    with_logprobs = []

    print(f"\nRunning alternating benchmark ({n_iterations} iterations per condition)...")
    print("Randomizing prompts to prevent caching effects...")

    # Create alternating sequence: without, with, without, with, ...
    # Total calls = 2 * n_iterations
    for i in range(n_iterations * 2):
        # Get a random prompt
        prompt_data = random.choice(PROMPT_POOL)
        messages = create_prompt(prompt_data["item"], prompt_data["response"])

        # Alternate between with and without logprobs
        use_logprobs = (i % 2 == 1)

        latency_ms = run_api_call(messages, use_logprobs=use_logprobs, top_logprobs=top_logprobs)

        if use_logprobs:
            with_logprobs.append(latency_ms)
            print(f"  {i+1:2d}. WITH logprobs:    {latency_ms:7.2f}ms ({prompt_data['item']})")
        else:
            without_logprobs.append(latency_ms)
            print(f"  {i+1:2d}. WITHOUT logprobs: {latency_ms:7.2f}ms ({prompt_data['item']})")

    return without_logprobs, with_logprobs

def print_statistics(latencies, label):
    """Print statistical summary of latencies."""
    mean = statistics.mean(latencies)
    median = statistics.median(latencies)
    stdev = statistics.stdev(latencies) if len(latencies) > 1 else 0
    min_val = min(latencies)
    max_val = max(latencies)

    print(f"\n{label} Statistics:")
    print(f"  Mean:   {mean:.2f}ms")
    print(f"  Median: {median:.2f}ms")
    print(f"  Std Dev: {stdev:.2f}ms")
    print(f"  Min:    {min_val:.2f}ms")
    print(f"  Max:    {max_val:.2f}ms")

    return mean, median, stdev

def main():
    print("OpenAI API Logprobs Latency Benchmark")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Iterations per condition: {N_ITERATIONS}")
    print("Method: Alternating conditions with randomized prompts")
    print("=" * 60)

    # Run alternating benchmark for top_logprobs=5
    latencies_without_5, latencies_with_5 = benchmark_alternating(top_logprobs=5)
    mean_without_5, median_without_5, _ = print_statistics(latencies_without_5, "WITHOUT logprobs (alternating)")
    mean_with_5, median_with_5, _ = print_statistics(latencies_with_5, "WITH logprobs (top_logprobs=5, alternating)")

    # Run alternating benchmark for top_logprobs=20 (if supported)
    print("\n" + "-" * 60)
    try:
        latencies_without_20, latencies_with_20 = benchmark_alternating(top_logprobs=20)
        mean_without_20, median_without_20, _ = print_statistics(latencies_without_20, "WITHOUT logprobs (alternating)")
        mean_with_20, median_with_20, _ = print_statistics(latencies_with_20, "WITH logprobs (top_logprobs=20, alternating)")
    except Exception as e:
        print(f"\nNote: top_logprobs=20 may not be supported: {e}")
        mean_with_20 = None
        mean_without_20 = None

    # Summary
    print("\n" + "=" * 60)
    print("OVERHEAD ANALYSIS")
    print("=" * 60)

    overhead_ms_5 = mean_with_5 - mean_without_5
    overhead_pct_5 = (overhead_ms_5 / mean_without_5) * 100

    print(f"\nLogprobs (top_logprobs=5) overhead:")
    print(f"  Baseline (without):  {mean_without_5:.2f}ms")
    print(f"  With logprobs:       {mean_with_5:.2f}ms")
    print(f"  Absolute overhead:   {overhead_ms_5:.2f}ms")
    print(f"  Relative overhead:   {overhead_pct_5:.1f}%")

    if mean_with_20 and mean_without_20:
        overhead_ms_20 = mean_with_20 - mean_without_20
        overhead_pct_20 = (overhead_ms_20 / mean_without_20) * 100
        print(f"\nLogprobs (top_logprobs=20) overhead:")
        print(f"  Baseline (without):  {mean_without_20:.2f}ms")
        print(f"  With logprobs:       {mean_with_20:.2f}ms")
        print(f"  Absolute overhead:   {overhead_ms_20:.2f}ms")
        print(f"  Relative overhead:   {overhead_pct_20:.1f}%")

    print("\nInterpretation:")
    if overhead_pct_5 < 10:
        print("  ✓ Overhead is minimal (<10%)")
    elif overhead_pct_5 < 20:
        print("  ~ Overhead is moderate (10-20%)")
    else:
        print("  ! Overhead is significant (>20%)")

    print("\n" + "=" * 60)
    print("Note: Alternating method controls for time-based effects.")
    print("Randomized prompts help prevent caching effects.")

if __name__ == "__main__":
    main()
