from enum import Enum
from dataclasses import dataclass,field
from typing import List, Dict,Tuple,Optional, Union
import numpy as np 

class OutcomeType(Enum):
    CORRECT = "Correct"
    NEAR_MISS = "NearMiss"
    FAILURE = "Failure"

@dataclass
class Node:
    node_id: int
    step_number: int
    step_embeddings: List[np.ndarray]
    avg_embedding: np.ndarray
    steps_text: List[str]
    is_greedy:bool
    is_leaf: bool = False
    outcome: Optional[OutcomeType] = None

@dataclass
class Edge:
    """Directed edge in reasoning graph"""
    from_node_id: int
    to_node_id: int
    step_text: str

@dataclass
class ReasoningGraph:
    """DAG representing multiple reasoning paths for one question"""
    question_id: int
    question_text: str
    ground_truth_answer: str
    nodes: Dict[int, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    adjacency_list: Dict[int, List[int]] = field(default_factory=dict)
    greedy_path: List[int] = field(default_factory=list)
    root_id: int = 0
    next_node_id: int = 1
    similarity_threshold: float = 0.9
    nearmiss_overlap_threshold: float = 0.7
    
    def __post_init__(self):
        """Initialize root node"""
        root = Node(
            node_id=0,
            step_number=0,
            step_embeddings=[],
            avg_embedding=np.zeros(384),  # adjust based on encoder dim
            steps_text=[]
        )
        self.nodes[0] = root
        self.adjacency_list[0] = []
        self.greedy_path = [0]