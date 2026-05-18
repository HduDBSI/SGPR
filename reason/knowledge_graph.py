"""
Knowledge Graph module for loading and querying graph data.
Implements Dijkstra-based shortest path search with score-aware ranking.
"""

import os
import json
import collections
import heapq
from typing import List, Tuple, Dict, Set


class KnowledgeGraph:
    """
    Knowledge Graph class that loads graph data and provides path-finding capabilities.
    """

    def __init__(self, base_kg_dir: str, dataset_name: str):
        """
        Initialize the Knowledge Graph.

        Args:
            base_kg_dir: Base directory containing KG data
            dataset_name: Name of the dataset (e.g., 'webqsp', 'cwq')
        """
        print("--- Loading Knowledge Graph ---")
        self.dir = os.path.join(base_kg_dir, dataset_name)
        self.id_to_name_map_path = os.path.join(self.dir, 'id_to_name_map.json')

        self.triples = self._load_json('kg_triples.json')
        self.entities = self._load_json('kg_entities.json')
        self.relations = self._load_json('kg_relations.json')

        self.id_to_name_map = {}
        if os.path.exists(self.id_to_name_map_path):
            with open(self.id_to_name_map_path, 'r', encoding='utf-8') as f:
                self.id_to_name_map = json.load(f)

        print(f"Knowledge Graph loaded from {self.dir}.")

        # Import normalize function from utils
        from utils import normalize
        self.entities_normalized = {normalize(self.resolve_id(ent_id)) for ent_id in self.entities}

        self._build_graph()

    def _load_json(self, filename: str):
        """Load JSON file from KG directory."""
        path = os.path.join(self.dir, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found at {path}. Please ensure KG files are in the correct dataset subdirectory.")
            exit()

    def resolve_id(self, entity_id_or_name: str) -> str:
        """
        Resolve entity ID to human-readable name.

        Args:
            entity_id_or_name: Entity ID or name

        Returns:
            Human-readable entity name
        """
        return self.id_to_name_map.get(entity_id_or_name, entity_id_or_name)

    def _build_graph(self):
        """Build adjacency list representation of the graph."""
        self.adj_list = collections.defaultdict(list)
        for h, r, t in self.triples:
            self.adj_list[h].append((r, t, [h, r, t]))
        print("Knowledge Graph adjacency list built.")

    def _get_triple_score(self, triple: List[str], triple_scores_map: Dict[Tuple[str, str, str], float]) -> float:
        """
        Get score for a triple from the score map.

        Args:
            triple: Triple as [head, relation, tail]
            triple_scores_map: Mapping from triple tuple to score

        Returns:
            Score for the triple (0.0 if not found)
        """
        return triple_scores_map.get(tuple(triple), 0.0)

    def find_shortest_paths_dijkstra(
            self,
            start_entity_id: str,
            max_hops: int,
            allowed_triples_set: Set[Tuple[str, str, str]],
            triple_scores_map: Dict[Tuple[str, str, str], float]
    ) -> List[Tuple[List[List[str]], float]]:
        """
        Find shortest paths from start entity using Dijkstra-like algorithm.
        Only traverses triples in allowed_triples_set.
        Path score is the sum of all triple scores in the path.

        Args:
            start_entity_id: Starting entity ID
            max_hops: Maximum number of hops
            allowed_triples_set: Set of allowed triples to traverse
            triple_scores_map: Mapping from triple to retrieval score

        Returns:
            List of (path, path_score_sum) tuples, where path is a list of triples
        """
        # Priority queue: (path_score_sum, hops, current_node_id, path_triples)
        pq = [(0.0, 0, start_entity_id, [])]

        distances = {start_entity_id: 0}
        best_path_score_sum_at_node = {start_entity_id: 0.0}
        shortest_path_to_entity = {start_entity_id: []}

        while pq:
            current_path_score_sum_neg, dist, current_node, current_path_triples = heapq.heappop(pq)
            current_path_score_sum = -current_path_score_sum_neg

            # Skip if we've found a better path
            if dist > distances.get(current_node, float('inf')) or \
                    (dist == distances.get(current_node, float('inf')) and
                     current_path_score_sum < best_path_score_sum_at_node.get(current_node, -float('inf'))):
                continue

            if dist >= max_hops:
                continue

            for rel_id, neighbor_id, full_triple in self.adj_list[current_node]:
                if tuple(full_triple) not in allowed_triples_set:
                    continue

                triple_score = self._get_triple_score(full_triple, triple_scores_map)
                new_path_score_sum = current_path_score_sum + triple_score
                new_dist = dist + 1

                # Update if we found a shorter path or better score at same distance
                if new_dist < distances.get(neighbor_id, float('inf')) or \
                        (new_dist == distances.get(neighbor_id, float('inf')) and
                         new_path_score_sum > best_path_score_sum_at_node.get(neighbor_id, -float('inf'))):
                    distances[neighbor_id] = new_dist
                    best_path_score_sum_at_node[neighbor_id] = new_path_score_sum
                    new_path_triples_for_neighbor = current_path_triples + [full_triple]
                    shortest_path_to_entity[neighbor_id] = new_path_triples_for_neighbor
                    heapq.heappush(pq, (-new_path_score_sum, new_dist, neighbor_id, new_path_triples_for_neighbor))

        # Collect unique paths with scores
        unique_shortest_paths_with_scores = []
        seen_path_tuples = set()

        for node_id, path in shortest_path_to_entity.items():
            if path:
                path_tuple = tuple(tuple(t) for t in path)
                if path_tuple not in seen_path_tuples:
                    unique_shortest_paths_with_scores.append((path, best_path_score_sum_at_node[node_id]))
                    seen_path_tuples.add(path_tuple)

        return unique_shortest_paths_with_scores