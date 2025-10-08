from typing import TypedDict, List, Dict, Optional, Annotated
import operator
from PIL import Image
import numpy as np

class PFDProcessingState(TypedDict):
    """Shared state across all agents in the workflow"""
    # Input
    image: Image.Image
    image_path: str
    
    # Vision analysis results
    detected_symbols: Annotated[List[Dict], operator.add]
    symbol_confidence: float
    
    # OCR results
    extracted_labels: Annotated[Dict[str, str], operator.add]
    text_regions: List[Dict]
    
    # Topology analysis
    connections: Annotated[List[Dict], operator.add]
    connection_confidence: float
    
    # Graph construction
    equipment_nodes: List[Dict]
    stream_edges: List[Dict]
    
    # Final output
    graph_data: Optional[Dict]
    networkx_graph: Optional[any]
    
    # Validation
    validation_passed: bool
    validation_errors: List[str]
    
    # Metadata
    processing_stage: str
    errors: Annotated[List[str], operator.add]
    warnings: Annotated[List[str], operator.add]