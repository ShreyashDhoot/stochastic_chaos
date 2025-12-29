def find_equivalent_node(graph: ReasoningGraph, prefix_avg_emb: np.ndarray,prefix_length: int) -> Optional[int]:
    """
    Find existing node with equivalent prefix
    Returns node_id if found, else None
    """
    for node_id, node in graph.nodes.items():
        if node.step_number == prefix_length:
            similarity = cosine_similarity(prefix_avg_emb, node.avg_embedding)
            if similarity >= graph.similarity_threshold:
                return node_id
    return None


def add_chain_to_graph(graph: ReasoningGraph,steps_text: List[str],step_embeddings: List[np.ndarray],outcome: OutcomeType,is_greedy: bool = False):
    """
    Add a single reasoning chain to the graph
    """
    current_node_id = graph.root_id
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
                steps_text=steps_text[:t].copy()
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
    
    # Mark final node as leaf
    graph.nodes[current_node_id].is_leaf = True
    graph.nodes[current_node_id].outcome = outcome
    
    # Track greedy path
    if is_greedy:
        # Reconstruct path from root to this leaf
        path = [graph.root_id]
        for t in range(1, len(steps_text) + 1):
            prefix_avg = np.mean(step_embeddings[:t], axis=0)
            node_id = find_equivalent_node(graph, prefix_avg, t)
            path.append(node_id if node_id else current_node_id)
        graph.greedy_path = path