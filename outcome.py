def extract_answer_from_generation(text: str) -> str:
    """Extract final answer from generated text"""
    # Look for common answer patterns
    patterns = [
        r'(?:final answer|answer|result).*?[:\s]+([^\n]+)',
        r'####\s*([^\n]+)',
        r'=\s*([0-9.]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Default: last number or last line
    nums = re.findall(r'\d+\.?\d*', text)
    return nums[-1] if nums else text.split('\n')[-1].strip()


def label_leaf_outcomes(graph: ReasoningGraph, gt_answer: str, gt_step_embs: List[np.ndarray] = None):
    """
    Universal outcome labeling:
    1. CORRECT = answer matches GT (primary!)
    2. NEAR_MISS = step similarity high (if gt_step_embs provided)
    3. FAILURE = otherwise
    """
    
    correct_nodes = []
    
    # STEP 1: Check answer correctness
    for node_id, node in graph.nodes.items():
        if not node.is_leaf or node.outcome is not None:
            continue
        
        # Extract answer from leaf's full text
        leaf_text = " ".join(node.steps_text)
        pred_answer = extract_answer_from_generation(leaf_text)
        
        # Check if matches GT
        if answers_match(pred_answer, gt_answer):
            node.outcome = OutcomeType.CORRECT
            correct_nodes.append(node)
            print(f"  ✓ Leaf {node_id}: CORRECT ({pred_answer} == {gt_answer})")
    
    print(f"Found {len(correct_nodes)} CORRECT leaves (answer match)")
    
    # STEP 2: NEAR_MISS via embeddings (if available)
    if gt_step_embs:
        for node_id, node in graph.nodes.items():
            if node.is_leaf and node.outcome is None:
                # Check embedding similarity
                step_sim = stepwise_cosine_similarity(node.step_embeddings, gt_step_embs)
                is_near = step_sim >= 0.7
                
                # Also check vs correct nodes
                if not is_near:
                    for correct_node in correct_nodes:
                        sim = stepwise_cosine_similarity(
                            node.step_embeddings, correct_node.step_embeddings
                        )
                        if sim >= 0.7:
                            is_near = True
                            break
                
                node.outcome = OutcomeType.NEAR_MISS if is_near else OutcomeType.FAILURE
    else:
        # No embeddings? All wrong answers = FAILURE
        for node_id, node in graph.nodes.items():
            if node.is_leaf and node.outcome is None:
                node.outcome = OutcomeType.FAILURE
    
    total_leaves = len([n for n in graph.nodes.values() if n.is_leaf])
    print(f"Labeled {total_leaves} leaves total")


def stepwise_cosine_similarity(path_embs: List[np.ndarray], gt_embs: List[np.ndarray]) -> float:
    """Mean cosine similarity of corresponding steps"""
    if not gt_embs:
        return 0.0
    
    min_len = min(len(path_embs), len(gt_embs))
    step_similarities = [
        np.dot(path_embs[i], gt_embs[i]) 
        for i in range(min_len)
    ]
    return np.mean(step_similarities) if step_similarities else 0.0
