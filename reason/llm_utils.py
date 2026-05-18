"""
LLM utility functions for generating answers using OpenAI API.
Includes retry logic and prompt formatting.
"""

import os
from typing import List, Tuple
from openai import OpenAI, APIConnectionError, APITimeoutError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import (
    OPENAI_MODEL_ID,
    OPENAI_API_BASE_URL,
    OPENAI_API_TIMEOUT,
    MAX_RETRY_ATTEMPTS,
    RETRY_MIN_WAIT,
    RETRY_MAX_WAIT,
    ICL_SYS_PROMPT_PATHS,
    ICL_USER_PROMPT_PATHS,
    ICL_ASS_PROMPT_PATHS,
    ICL_COT_PROMPT
)


def format_path_for_llm(path_triples: List[List[str]], kg) -> str:
    """
    Format a path (list of triples) as a string for LLM prompt.

    Args:
        path_triples: List of triples [head, relation, tail]
        kg: KnowledgeGraph instance for ID resolution

    Returns:
        Formatted path string
    """
    formatted_triples = []
    for h_id, p_id, o_id in path_triples:
        h_name = kg.resolve_id(h_id)
        o_name = kg.resolve_id(o_id)
        formatted_triples.append(f"({h_name},{p_id},{o_name})")
    return " -> ".join(formatted_triples)


def format_triples_for_llm(triples_with_scores: List[Tuple[List[str], float]], kg) -> str:
    """
    Format triples with scores as a string for LLM prompt.
    Note: Scores are not included in the formatted output.

    Args:
        triples_with_scores: List of (triple, score) tuples
        kg: KnowledgeGraph instance for ID resolution

    Returns:
        Formatted triples string
    """
    formatted_triples = []
    for triple, score in triples_with_scores:
        h_name = kg.resolve_id(triple[0])
        o_name = kg.resolve_id(triple[2])
        formatted_triples.append(f"({h_name},{triple[1]},{o_name})")
    return "\n".join(formatted_triples)


@retry(
    stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
    retry=retry_if_exception_type((APIConnectionError, APITimeoutError))
)
def generate_with_subgraphrag_logic(
        question: str,
        data_for_llm: List[Tuple[List[List[str]], float]],
        kg
) -> str:
    """
    Generate answer using LLM with path-based reasoning.
    Implements retry logic for API failures.

    Args:
        question: Question to answer
        data_for_llm: List of (path, score) tuples
        kg: KnowledgeGraph instance

    Returns:
        LLM-generated answer string

    Raises:
        ValueError: If API key is not set or API response is invalid
        APIConnectionError: If API connection fails (triggers retry)
        APITimeoutError: If API request times out (triggers retry)
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set.")

    client = OpenAI(api_key=api_key, base_url=OPENAI_API_BASE_URL, timeout=OPENAI_API_TIMEOUT)

    # Format paths for prompt
    path_prompt_lines = []
    for path_list, score in data_for_llm:
        formatted_path = format_path_for_llm(path_list, kg)
        path_prompt_lines.append(f"Path: {formatted_path}")

    input_content = "No relevant paths found." if not path_prompt_lines else "\n".join(path_prompt_lines)
    user_query_content = f"Paths:\n{input_content}\n\nQuestion:\n{question}"

    # Build messages with few-shot examples
    messages = [
        {"role": "system", "content": ICL_SYS_PROMPT_PATHS},
        {"role": "user", "content": ICL_USER_PROMPT_PATHS},
        {"role": "assistant", "content": ICL_ASS_PROMPT_PATHS},
        {"role": "user", "content": user_query_content}
    ]

    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL_ID,
            messages=messages,
            temperature=0.0,
            max_tokens=1024
        )

        if not hasattr(completion, 'choices') or not completion.choices:
            raise ValueError(f"Unexpected API response format: {completion}")

        initial_prediction = completion.choices[0].message.content.strip()

        # If no answer found, trigger Chain-of-Thought reasoning
        if 'ans:' not in initial_prediction.lower() or "ans: not available" in initial_prediction.lower():
            messages.append({"role": "assistant", "content": initial_prediction})
            messages.append({"role": "user", "content": ICL_COT_PROMPT})

            completion_dc = client.chat.completions.create(
                model=OPENAI_MODEL_ID,
                messages=messages,
                temperature=0.0,
                max_tokens=1024
            )

            if not hasattr(completion_dc, 'choices') or not completion_dc.choices:
                raise ValueError(f"Unexpected API response format in CoT step: {completion_dc}")

            return completion_dc.choices[0].message.content.strip()
        else:
            return initial_prediction

    except (APIConnectionError, APITimeoutError) as e:
        print(f"OpenAI API request failed for question '{question}': {e}. Retrying...")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during API call for question '{question}': {e}. Retrying...")
        raise