"""
Evaluation metrics for RAG pipeline.
Includes F1, Hit@1, and hallucination score calculations.
"""

import string
import re
from copy import deepcopy
from typing import List, Set


def normalize(s: str) -> str:
    """
    Normalize a string by lowercasing, removing punctuation and articles.

    Args:
        s: Input string

    Returns:
        Normalized string
    """
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    """
    Check if two strings match (one contains the other).

    Args:
        s1: First string
        s2: Second string

    Returns:
        True if strings match
    """
    return s2 in s1 or s1 in s2


def get_pred(prediction_str: str) -> List[str]:
    """
    Extract predicted entities from LLM output string.

    Args:
        prediction_str: Raw LLM output

    Returns:
        List of normalized predicted entities
    """
    res = [p for p in prediction_str.split("\n") if 'ans:' in p.lower() and 'none' not in p.lower()]
    if res:
        res = [p for p in res if "ans: not available" not in p.lower()]

    cleaned_res = [normalize(p.split('ans:', 1)[-1].strip()) for p in res]

    # Remove duplicates while preserving order
    seen, unique_list = set(), []
    for item in cleaned_res:
        if item not in seen:
            unique_list.append(item)
            seen.add(item)

    return unique_list


def get_all_retrieved_entities_from_paths(paths_with_scores: List[tuple], kg) -> List[str]:
    """
    Extract all entities from paths or triples and normalize them.

    Args:
        paths_with_scores: List of (path/triple, score) tuples
        kg: KnowledgeGraph instance

    Returns:
        List of normalized entity names
    """
    all_ent = set()
    for item, score in paths_with_scores:
        # Check if item is a single triple (list of 3 strings)
        if len(item) == 3 and isinstance(item[0], str):
            triple = item
            all_ent.add(normalize(kg.resolve_id(triple[0])))
            all_ent.add(normalize(kg.resolve_id(triple[2])))
        else:  # Assume it's a path (list of triples)
            for triple in item:
                all_ent.add(normalize(kg.resolve_id(triple[0])))
                all_ent.add(normalize(kg.resolve_id(triple[2])))

    return list(all_ent)


def calculate_f1_metrics(prediction_list: List[str], answer_list: List[str]):
    """
    Calculate F1, precision, and recall metrics.

    Args:
        prediction_list: List of predicted entities
        answer_list: List of ground truth entities

    Returns:
        Tuple of (f1, precision, recall)
    """
    if not answer_list:
        return (1.0, 1.0, 1.0) if not prediction_list else (0.0, 0.0, 0.0)
    if not prediction_list:
        return 0.0, 0.0, 0.0

    prediction_set, answer_set = set(prediction_list), set(answer_list)
    matched = sum(1 for pred_item in prediction_set if any(match(pred_item, ans_item) for ans_item in answer_set))

    precision = matched / len(prediction_set) if prediction_set else 0
    recall = matched / len(answer_set) if answer_set else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1, precision, recall


def eval_hit(prediction_list: List[str], answer_list: List[str]) -> int:
    """
    Calculate Hit@1 metric (whether top prediction is correct).

    Args:
        prediction_list: List of predicted entities
        answer_list: List of ground truth entities

    Returns:
        1 if hit, 0 otherwise
    """
    if not prediction_list:
        return 0
    return 1 if any(match(prediction_list[0], a) for a in answer_list) else 0


def eval_hal_score(
    prediction_list: List[str],
    answer_list: List[str],
    good_sample: bool,
    no_ans_flag: bool,
    subgraph_ent_list: List[str]
) -> float:
    """
    Calculate hallucination score for a single sample.

    Score ranges from -1.5 (worst) to 1.0 (best):
    - For good samples (ground truth in KG):
      - Correct predictions: +1 per entity
      - Wrong predictions: -1 per entity
      - No answer when answer exists: 0
    - For bad samples (ground truth not in KG):
      - No answer: +1
      - Predictions in subgraph: -1 per entity
      - Predictions outside subgraph: -1.5 per entity

    Args:
        prediction_list: List of predicted entities
        answer_list: List of ground truth entities
        good_sample: Whether ground truth entities exist in KG
        no_ans_flag: Whether model returned "not available"
        subgraph_ent_list: List of entities in retrieved subgraph

    Returns:
        Hallucination score (normalized by number of predictions)
    """
    answer, prediction = deepcopy(answer_list), deepcopy(prediction_list)
    score = 0

    if len(prediction) == 0 and not no_ans_flag:
        no_ans_flag = True

    if good_sample:
        if no_ans_flag:
            return 0.0

        for pred_item in prediction:
            matched = False
            for a in answer:
                if match(pred_item, a):
                    score += 1
                    matched = True
                    answer.remove(a)
                    break

            if not matched:
                score -= 1

        return score / len(prediction) if prediction else 0.0
    else:
        if no_ans_flag:
            return 1.0

        for pred_item in prediction:
            if any(match(pred_item, ent) for ent in subgraph_ent_list):
                score -= 1
            else:
                score -= 1.5

        return score / len(prediction) if prediction else 0.0