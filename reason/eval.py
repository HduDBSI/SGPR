"""
Evaluation script for analyzing RAG pipeline results from log files.
Calculates aggregate metrics including F1, Hit@1, Exact Match, and Hallucination Score.
"""

import os
import json
import argparse
from evaluation import normalize, eval_hal_score


def evaluate_from_log(log_file_path: str):
    """
    Load an evaluation log and calculate aggregate metrics.

    Args:
        log_file_path: Path to the JSONL log file
    """
    if not os.path.exists(log_file_path):
        print(f"Error: Log file not found at {log_file_path}")
        return

    # Load evaluation log
    evaluation_log = []
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        evaluation_log.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse JSON line {line_num}: {e}")
                        continue
    except Exception as e:
        print(f"Error reading log file {log_file_path}: {e}")
        return

    if not evaluation_log:
        print("No valid samples found in the log file to evaluate.")
        return

    # Filter successfully processed samples
    actual_processed_samples = [
        item for item in evaluation_log
        if item.get('status') == 'processed_successfully'
    ]
    num_actual_processed = len(actual_processed_samples)

    if num_actual_processed == 0:
        print("No successfully processed samples found in the log file to evaluate.")
        return

    # Initialize aggregate metrics
    totals = {
        'f1': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'hal_score': 0.0,
        'hit1': 0.0,
        'em': 0.0,
        'tw': 0.0
    }
    micro_totals = {'tp': 0, 'pred_count': 0, 'gt_count': 0}

    # Process each sample
    for item in actual_processed_samples:
        metrics = item['metrics']

        # Extract data for recalculation
        predicted_entities = item.get('prediction_parsed', [])
        ground_truth_entities = {normalize(ent) for ent in item.get('ground_truth', [])}
        good_sample = item.get('good_sample', False)
        no_ans_flag = item.get('no_ans_flag', False)
        subgraph_ent_list = item.get('subgraph_ent_list', [])

        # Recalculate Hal Score
        recalculated_hal_score = eval_hal_score(
            predicted_entities,
            list(ground_truth_entities),
            good_sample,
            no_ans_flag,
            subgraph_ent_list
        )

        # Aggregate metrics
        totals['f1'] += metrics['f1']
        totals['precision'] += metrics['precision']
        totals['recall'] += metrics['recall']
        totals['hit1'] += metrics['hit1']
        totals['hal_score'] += recalculated_hal_score

        # Calculate EM and TW
        totals['em'] += (1 if set(predicted_entities) == ground_truth_entities else 0)
        if metrics['recall'] == 0:
            totals['tw'] += 1

        # Aggregate micro F1 components
        micro_totals['tp'] += len(set(predicted_entities).intersection(ground_truth_entities))
        micro_totals['pred_count'] += len(predicted_entities)
        micro_totals['gt_count'] += len(ground_truth_entities)

    # Calculate final average metrics
    final_scores = {k: v / num_actual_processed for k, v in totals.items()}

    # Calculate micro F1
    micro_p = micro_totals['tp'] / micro_totals['pred_count'] if micro_totals['pred_count'] > 0 else 0
    micro_r = micro_totals['tp'] / micro_totals['gt_count'] if micro_totals['gt_count'] > 0 else 0
    micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

    # Normalize Hal Score to 0-100 scale
    normalized_hal = (final_scores['hal_score'] - (-1.5)) / (1.0 - (-1.5)) * 100

    # Print report
    print("\n" + "=" * 80)
    print("--- AGGREGATE RAG PIPELINE EVALUATION REPORT ---")
    print(f"Log File: {log_file_path}")
    print("-" * 80)
    print(f"Total Samples Processed: {num_actual_processed}")
    print("\nOverall Metrics:")
    print(
        f"  Macro F1:      {final_scores['f1']:.4f} (P: {final_scores['precision']:.4f}, R: {final_scores['recall']:.4f})")
    print(f"  Micro F1:      {micro_f1:.4f} (P: {micro_p:.4f}, R: {micro_r:.4f})")
    print(f"  Hit@1:         {final_scores['hit1']:.4f}")
    print(f"  Exact Match:   {final_scores['em']:.4f}")
    print(f"  Hal Score:     {normalized_hal:.2f} (Normalized 0-100)")
    print(f"  Total Zero Recall: {int(totals['tw'])} samples")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate aggregate metrics from RAG pipeline evaluation log."
    )
    parser.add_argument(
        '--log-file',
        type=str,
        required=True,
        help="Path to the evaluation log JSONL file"
    )
    args = parser.parse_args()
    evaluate_from_log(args.log_file)