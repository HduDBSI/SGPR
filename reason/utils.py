"""
Utility functions for file I/O, logging, and result management.
"""

import os
import json
from typing import Set, Tuple


def load_existing_results_and_metrics(log_file_path: str) -> Tuple[Set[str], float, float, int]:
    """
    Load results from an existing JSONL log file and recalculate metrics.
    Used for resuming interrupted experiments.

    Args:
        log_file_path: Path to the JSONL log file

    Returns:
        Tuple of (processed_qids, running_f1_sum, running_hit1_sum, successful_processed_count)
    """
    processed_qids = set()
    running_f1_sum = 0.0
    running_hit1_sum = 0.0
    successful_processed_count = 0

    if not os.path.exists(log_file_path):
        return processed_qids, running_f1_sum, running_hit1_sum, successful_processed_count

    print(f"Attempting to resume from existing log file: {log_file_path}")
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        q_id = item.get('id')
                        if q_id:
                            processed_qids.add(q_id)
                            if item.get('status') == 'processed_successfully':
                                metrics = item.get('metrics', {})
                                running_f1_sum += metrics.get('f1', 0.0)
                                running_hit1_sum += metrics.get('hit1', 0.0)
                                successful_processed_count += 1
                        else:
                            print(f"Warning: Line {line_num} in '{log_file_path}' has no 'id' field. Skipping.")
                    except json.JSONDecodeError as e:
                        print(
                            f"Warning: Could not parse JSON line {line_num} in '{log_file_path}': {line.strip()} - Error: {e}. Skipping this line.")
                        continue
    except Exception as e:
        print(f"Error reading existing log file '{log_file_path}': {e}. Starting fresh.")
        return set(), 0.0, 0.0, 0

    print(f"Found {len(processed_qids)} previously processed samples.")
    if successful_processed_count > 0:
        print(f"  Initial Avg F1: {running_f1_sum / successful_processed_count:.3f}")
        print(f"  Initial Avg Hit@1: {running_hit1_sum / successful_processed_count:.3f}")

    return processed_qids, running_f1_sum, running_hit1_sum, successful_processed_count


def generate_final_report(args, running_f1_sum: float, running_hit1_sum: float, successful_processed_count: int):
    """
    Generate and print final evaluation report.

    Args:
        args: Command line arguments
        running_f1_sum: Cumulative F1 score
        running_hit1_sum: Cumulative Hit@1 score
        successful_processed_count: Number of successfully processed samples
    """
    print("\n" + "=" * 80)
    if args.target_id:
        print(f"--- RAG PIPELINE EVALUATION REPORT (Single Sample ID: {args.target_id}) ---")
    else:
        print("--- RAG PIPELINE EVALUATION REPORT (UNIFIED STRATEGY) ---")

    print(f"Strategy: Unified Path Search (No Classification)")
    print(f"Max Hops: {args.unified_max_hops}")
    print(f"Top Paths to LLM: {args.unified_top_paths}")

    from config import OPENAI_MODEL_ID
    print(f"LLM Reasoner: {OPENAI_MODEL_ID}")
    print("-" * 80)

    print(f"Total Samples Selected for Processing: {len(args.final_selected_sample_ids)}")
    print(f"Successfully Processed Samples (contributing to averages): {successful_processed_count}")

    if successful_processed_count > 0:
        final_avg_f1 = running_f1_sum / successful_processed_count
        final_avg_hit1 = running_hit1_sum / successful_processed_count
        print(f"Overall Averages:")
        print(f"  Final Avg F1: {final_avg_f1:.3f}")
        print(f"  Final Avg Hit@1: {final_avg_hit1:.3f}")
    else:
        print("No samples were successfully processed to calculate overall averages.")

    print(f"\nDetailed transparent log for each sample saved to: {os.path.abspath(args.output_file)}")
    print("=" * 80)