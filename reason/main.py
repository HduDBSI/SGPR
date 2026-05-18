"""
Main evaluation script for RAG pipeline with unified path search strategy.
"""

import os
import json
import time
import argparse
import torch
from tqdm import tqdm
from datasets import load_dataset

from config import KG_BASE_DIR, DATASET_REPO_MAP
from knowledge_graph import KnowledgeGraph
from llm_utils import generate_with_subgraphrag_logic
from evaluation import (
    normalize,
    get_pred,
    get_all_retrieved_entities_from_paths,
    calculate_f1_metrics,
    eval_hit,
    eval_hal_score
)
from utils import load_existing_results_and_metrics, generate_final_report


def main(args):
    """
    Main evaluation function.

    Args:
        args: Command line arguments
    """
    # Load Knowledge Graph
    kg = KnowledgeGraph(KG_BASE_DIR, args.dataset)

    # Load pre-retrieved data
    all_pre_retrieved_data = torch.load(args.pre_retrieved_results_path)
    print(f"Loaded question-specific pre-retrieved data from {args.pre_retrieved_results_path}.")
    preretrieved_qids = set(all_pre_retrieved_data.keys())
    print(f"Found {len(preretrieved_qids)} unique question IDs in local data.")

    # Fetch gold-standard IDs from Hugging Face
    repo_id = DATASET_REPO_MAP[args.dataset]
    print(f"Fetching gold-standard IDs from Hugging Face: {repo_id} (test split)")
    hf_ds = load_dataset(repo_id, split='test')
    hf_test_ids = [str(item['id']) for item in hf_ds]
    print(f"Found {len(hf_test_ids)} IDs in Hugging Face test set.")

    # Determine final samples to process
    if args.target_id:
        if args.target_id not in preretrieved_qids:
            print(f"Error: Target ID {args.target_id} not found in local data.")
            return
        initial_selected_sample_ids = [args.target_id]
    else:
        # Filter by HF test IDs and maintain order
        initial_selected_sample_ids = [q for q in hf_test_ids if q in preretrieved_qids]

        # Apply sample limit if specified
        if args.num_samples != float('inf'):
            initial_selected_sample_ids = initial_selected_sample_ids[:int(args.num_samples)]

    print(f"Final selected {len(initial_selected_sample_ids)} samples for evaluation.")

    initial_selected_sample_ids.sort()
    args.final_selected_sample_ids = initial_selected_sample_ids

    # Setup output file
    if args.output_file is None:
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        args.output_file = os.path.join(output_dir, f'{args.dataset}_unified_results.jsonl')

    # Load existing results for resumption
    processed_qids_from_file, running_f1_sum, running_hit1_sum, successful_processed_count = \
        load_existing_results_and_metrics(args.output_file)

    final_selected_sample_ids = [q for q in initial_selected_sample_ids if q not in processed_qids_from_file]

    pbar = tqdm(
        final_selected_sample_ids,
        desc="Evaluating RAG (Unified)",
        total=len(initial_selected_sample_ids),
        initial=successful_processed_count
    )

    file_open_mode = 'a' if os.path.exists(args.output_file) and len(processed_qids_from_file) > 0 else 'w'

    with open(args.output_file, file_open_mode, encoding='utf-8') as outfile:
        for question_id in pbar:
            current_sample_data = all_pre_retrieved_data.get(question_id)
            if current_sample_data is None:
                continue

            question = current_sample_data.get('question', "N/A")
            ground_truth_raw = set(current_sample_data.get('a_entity', []))
            topic_entities = current_sample_data.get('q_entity', [])
            current_question_scored_triples_raw = current_sample_data.get('scored_triples', [])

            if not current_question_scored_triples_raw:
                continue

            complexity = 'unified'

            # Unified strategy: prepare data
            sorted_retrieved_triples = sorted(
                current_question_scored_triples_raw,
                key=lambda x: x[3],
                reverse=True
            )
            top_n_triples = sorted_retrieved_triples

            print(f"\n[ID: {question_id}] Question: '{question}' -> Strategy: UNIFIED PATH SEARCH")

            # Prepare Dijkstra data structures
            allowed_triples_set = {tuple(t[:3]) for t in top_n_triples}
            triple_scores_map = {tuple(t[:3]): t[3] for t in top_n_triples}

            # Execute path search
            all_found_paths_with_scores = []
            for topic_ent_id in topic_entities:
                if topic_ent_id in kg.adj_list:
                    paths = kg.find_shortest_paths_dijkstra(
                        topic_ent_id,
                        args.max_hops,
                        allowed_triples_set,
                        triple_scores_map
                    )
                    all_found_paths_with_scores.extend(paths)

            # Calculate average scores and deduplicate
            unique_paths_by_avg_map = {}
            for path_list, sum_score in all_found_paths_with_scores:
                path_tuple = tuple(tuple(t) for t in path_list)
                path_len = len(path_list)
                avg_score = sum_score / path_len if path_len > 0 else 0.0

                if path_tuple not in unique_paths_by_avg_map or avg_score > unique_paths_by_avg_map[path_tuple][1]:
                    unique_paths_by_avg_map[path_tuple] = (path_list, avg_score)

            # Sort by average score and select top-K
            sorted_unique_paths = sorted(
                list(unique_paths_by_avg_map.values()),
                key=lambda x: x[1],
                reverse=True
            )
            final_data_for_llm = sorted_unique_paths[:args.top_paths]

            print(f" -> Selected top {len(final_data_for_llm)} paths based on AVERAGE Score.")

            # Generate answer
            start_time = time.time()
            predicted_answer_str = generate_with_subgraphrag_logic(
                question,
                final_data_for_llm,
                kg
            )
            end_time = time.time()
            generation_time = end_time - start_time

            # Calculate metrics
            predicted_entities = get_pred(predicted_answer_str)
            ground_truth_entities = {normalize(ent) for ent in ground_truth_raw}
            f1, precision, recall = calculate_f1_metrics(predicted_entities, list(ground_truth_entities))
            hit1 = eval_hit(predicted_entities, list(ground_truth_entities))

            good_sample = any(ent in kg.entities_normalized for ent in ground_truth_entities)
            no_ans_flag = "ans: not available" in predicted_answer_str.lower() or not predicted_entities
            subgraph_ent_list = get_all_retrieved_entities_from_paths(final_data_for_llm, kg)
            hal_score = eval_hal_score(
                predicted_entities,
                list(ground_truth_entities),
                good_sample,
                no_ans_flag,
                subgraph_ent_list
            )

            # Save result
            sample_result = {
                'id': question_id,
                'question': question,
                'complexity': complexity,
                'final_data_for_llm': final_data_for_llm,
                'ground_truth': list(ground_truth_raw),
                'prediction_raw': predicted_answer_str,
                'prediction_parsed': predicted_entities,
                'good_sample': good_sample,
                'no_ans_flag': no_ans_flag,
                'subgraph_ent_list': subgraph_ent_list,
                'time_taken_seconds': round(generation_time, 4),
                'metrics': {
                    'f1': f1,
                    'hit1': hit1,
                    'precision': precision,
                    'recall': recall,
                    'hal_score': hal_score
                },
                'status': 'processed_successfully'
            }
            outfile.write(json.dumps(sample_result, ensure_ascii=False) + '\n')

            running_f1_sum += f1
            running_hit1_sum += hit1
            successful_processed_count += 1

            pbar.set_postfix({'Avg F1': f"{running_f1_sum / successful_processed_count:.3f}"})

    generate_final_report(args, running_f1_sum, running_hit1_sum, successful_processed_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate RAG with Unified Strategy.")

    parser.add_argument(
        '--num-samples',
        type=lambda x: int(x) if x.isdigit() else float('inf'),
        default=float('inf'),
        help="Number of samples to process (default: all)"
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help="Output JSONL file path"
    )
    parser.add_argument(
        '-d', '--dataset',
        type=str,
        required=True,
        choices=['webqsp', 'cwq'],
        help="Dataset name"
    )
    parser.add_argument(
        '--pre-retrieved-results-path',
        type=str,
        required=True,
        help="Path to pre-retrieved results file"
    )
    parser.add_argument(
        '--target-id',
        type=str,
        default=None,
        help="Process only a specific question ID"
    )
    parser.add_argument(
        '--load-sample-ids-from',
        type=str,
        default=None,
        help="Load sample IDs from file"
    )
    parser.add_argument(
        '--max-hops',
        type=int,
        default=2,
        help="Max hops for unified shortest path search"
    )
    parser.add_argument(
        '--top-paths',
        type=int,
        default=128,
        help="Number of top paths (ranked by Avg Score) to send to LLM"
    )

    args = parser.parse_args()
    main(args)