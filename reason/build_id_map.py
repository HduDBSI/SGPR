"""
Build knowledge graph files from Hugging Face datasets.
Generates kg_entities.json, kg_relations.json, kg_triples.json, and id_to_name_map.json.
"""

import os
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets


def build_kg_files(base_kg_dir: str, dataset_name: str):
    """
    Extract and generate KG files from Hugging Face dataset.

    Args:
        base_kg_dir: Base directory for KG files
        dataset_name: Dataset name ('webqsp' or 'cwq')
    """
    print(f"--- Building KG files for {dataset_name} ---")

    # Create dataset-specific directory
    dataset_kg_dir = os.path.join(base_kg_dir, dataset_name)
    os.makedirs(dataset_kg_dir, exist_ok=True)

    all_entities = set()
    all_relations = set()
    all_triples_raw = set()
    id_to_name_map = {}

    # Load dataset
    hf_dataset_name = 'rmanluo/RoG-webqsp' if dataset_name == 'webqsp' else 'rmanluo/RoG-cwq'
    full_dataset = concatenate_datasets([
        load_dataset(hf_dataset_name, split="train"),
        load_dataset(hf_dataset_name, split="validation"),
        load_dataset(hf_dataset_name, split="test")
    ])

    # Extract entities, relations, and triples
    for item in tqdm(full_dataset, desc=f"Processing {dataset_name}"):
        graph_triples = item.get('graph', [])

        for triple_list in graph_triples:
            if isinstance(triple_list, list) and len(triple_list) == 3:
                s, p, o = [str(x).strip() for x in triple_list]
                all_entities.add(s)
                all_relations.add(p)
                all_entities.add(o)
                all_triples_raw.add(tuple(triple_list))

        # Build id_to_name_map
        q_entities = item.get('q_entity', [])
        a_entities = item.get('a_entity', [])
        human_names = set(q_entities + a_entities)

        for triple_list in graph_triples:
            if isinstance(triple_list, list) and len(triple_list) == 3:
                s, p, o = [str(x).strip() for x in triple_list]
                if s.startswith("m.") and o in human_names and s != o:
                    id_to_name_map[s] = o
                if o.startswith("m.") and s in human_names and s != o:
                    id_to_name_map[o] = s

    # Scan for explicit name relations
    name_relations = [
        "type.object.name", "common.topic.alias", "base.type_ontology.name",
        "people.person.name", "organization.organization.name", "location.location.name",
        "film.film.name", "tv.tv_program.name", "book.book.name"
    ]

    for h, p, o in tqdm(all_triples_raw, desc="Scanning for ID-Name relations"):
        if h.startswith("m.") and p in name_relations and isinstance(o, str) and not o.startswith("m."):
            id_to_name_map[h] = o
        elif o.startswith("m.") and p in name_relations and isinstance(h, str) and not h.startswith("m."):
            id_to_name_map[o] = h

    # Save files
    with open(os.path.join(dataset_kg_dir, 'kg_entities.json'), 'w', encoding='utf-8') as f:
        json.dump(sorted(list(all_entities)), f, indent=4, ensure_ascii=False)
    print(f"✓ kg_entities.json saved")

    with open(os.path.join(dataset_kg_dir, 'kg_relations.json'), 'w', encoding='utf-8') as f:
        json.dump(sorted(list(all_relations)), f, indent=4, ensure_ascii=False)
    print(f"✓ kg_relations.json saved")

    with open(os.path.join(dataset_kg_dir, 'kg_triples.json'), 'w', encoding='utf-8') as f:
        json.dump([list(t) for t in all_triples_raw], f, indent=4, ensure_ascii=False)
    print(f"✓ kg_triples.json saved")

    with open(os.path.join(dataset_kg_dir, 'id_to_name_map.json'), 'w', encoding='utf-8') as f:
        json.dump(id_to_name_map, f, indent=4, ensure_ascii=False)
    print(f"✓ id_to_name_map.json saved")

    print(f"\nFinished building KG files in {dataset_kg_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Build KG files from Hugging Face datasets."
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['webqsp', 'cwq'],
        help="Dataset name"
    )
    parser.add_argument(
        '--kg-base-dir',
        type=str,
        default='data/kg_final',
        help="Base directory for KG files"
    )
    args = parser.parse_args()

    build_kg_files(args.kg_base_dir, args.dataset)