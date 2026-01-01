from typing import List, Optional
import numpy as np
from graph_structure import ReasoningGraph, Node, Edge, OutcomeType

def cosine_similarity_vec(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def find_equivalent_node(graph: ReasoningGraph, prefix_avg_emb: np.ndarray,prefix_length: int) -> Optional[int]:
    """
    Find existing node with equivalent prefix
    Returns node_id if found, else None
    """
    for node_id, node in graph.nodes.items():
        if node.step_number == prefix_length:
            similarity = cosine_similarity_vec(prefix_avg_emb, node.avg_embedding)
            if similarity >= graph.similarity_threshold:
                return node_id
    return None


def add_chain_to_graph(graph: ReasoningGraph,steps_text: List[str],step_embeddings: List[np.ndarray],is_greedy: bool = False,logprob: float | None = None,): 
    """
    Add a single reasoning chain to the graph
    """
    current_node_id = graph.root_id
    traversed_path=[graph.root_id]
    accumulated_embeddings = []
    
    for t in range(1, len(steps_text) + 1):
        step_text = steps_text[t - 1]
        step_emb = step_embeddings[t - 1]
        
        # Accumulate embeddings for prefix of length t
        accumulated_embeddings.append(step_emb)
        prefix_avg_emb = np.mean(accumulated_embeddings, axis=0)
        
        # Check if equivalent node exists
        equivalent_node_id = find_equivalent_node(graph, prefix_avg_emb, t)
        
        if equivalent_node_id is not None:
            # Reuse existing node
            next_node_id = equivalent_node_id
        else:
            # Create new node
            next_node_id = graph.next_node_id
            graph.next_node_id += 1
            
            new_node = Node(
                node_id=next_node_id,
                step_number=t,
                step_embeddings=accumulated_embeddings.copy(),
                avg_embedding=prefix_avg_emb,
                steps_text=steps_text[:t].copy(),
                is_greedy=is_greedy
            )
            graph.nodes[next_node_id] = new_node
            graph.adjacency_list[next_node_id] = []
        
        # Create edge 
        if next_node_id not in graph.adjacency_list[current_node_id]:
            edge = Edge(
                from_node_id=current_node_id,
                to_node_id=next_node_id,
                step_text=step_text
            )
            graph.edges.append(edge)
            graph.adjacency_list[current_node_id].append(next_node_id)
        
        current_node_id = next_node_id
        traversed_path.append(next_node_id)
    
    # Mark final node as leaf
    leaf_id = current_node_id
    graph.nodes[current_node_id].is_leaf = True
    
        # store / merge leaf logprob (MAX over merged chains)
    if logprob is not None:
        if not hasattr(graph, "leaf_logprobs"):
            graph.leaf_logprobs = {}  # Dict[int, float]
        graph.leaf_logprobs[leaf_id] = max(logprob, graph.leaf_logprobs.get(leaf_id, float("-inf")))
    
    # Track greedy path
    if is_greedy:
        graph.greedy_path = traversed_path