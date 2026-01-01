import re
from typing import List, Tuple

def extract_final_answer(answer_text: str, dataset_name: str) -> Tuple[str, List[str]]:
    """
    Extract final answer + reasoning steps
    Returns: (final_answer, solution_steps)
    """
    
    # GSM8K format: "#### 72"
    if "####" in answer_text:
        parts = answer_text.split("####")
        solution_steps = [s.strip() for s in parts[0].split("\n") if s.strip()]
        final_answer = parts[1].strip()
        return final_answer, solution_steps
    
    # Binary/Short answer format: "145" or "Yes/No"
    if len(answer_text.split("\n")) <= 3:
        lines = [l.strip() for l in answer_text.split("\n") if l.strip()]
        final_answer = lines[-1]  # Last line = answer
        solution_steps = lines[:-1] if len(lines) > 1 else []
        return final_answer, solution_steps
    
    # Expression format: "( 290.0 / 2.0 )\n145"
    expr_match = re.search(r'\([^)]+\)', answer_text)
    if expr_match:
        solution_steps = [expr_match.group(0)]
        # Extract final number
        nums = re.findall(r'\d+\.?\d*', answer_text)
        final_answer = nums[-1] if nums else answer_text.strip()
        return final_answer, solution_steps
    
    # Default: Last line = answer, rest = steps
    lines = [l.strip() for l in answer_text.split("\n") if l.strip()]
    final_answer = lines[-1] if lines else answer_text
    solution_steps = lines[:-1]
    
    return final_answer, solution_steps


def normalize_answer(answer: str) -> str:
    """Normalize for comparison"""
    # Convert to string if not already (safety for bool types)
    if not isinstance(answer, str):
        answer = str(answer).lower()
    
    answer = answer.lower().strip()
    
    # Boolean answers (StrategyQA - check first before regex)
    if answer in ['true', 'false']:
        return answer

    # Extract numbers
    nums = re.findall(r'-?\d+\.?\d*', answer)
    if nums:
        return nums[-1]  # Last number
    
    # Clean text
    return answer.lower().strip().replace(",", "")


def answers_match(pred: str, gt: str, threshold: float = 0.95) -> bool:
    """Fuzzy answer matching"""
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)
    
    # Exact match
    if pred_norm == gt_norm:
        return True
    
    # Numeric match (within 1%)
    try:
        pred_num = float(pred_norm)
        gt_num = float(gt_norm)
        if abs(pred_num - gt_num) / max(abs(gt_num), 1e-6) < 0.01:
            return True
    except:
        pass
    
    return False
