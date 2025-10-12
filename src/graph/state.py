from typing import TypedDict, List, Dict, Optional, Annotated
import operator
from PIL import Image
import numpy as np
from pydantic import BaseModel, Field
import logger 
from models.equipment import Equipment
class PFDProcessingState(TypedDict):
    """Shared state across all agents in the workflow"""
    # Input
    image: Image.Image
    image_path: str
    
    # Vision analysis results
    detected_symbols: Annotated[List[Dict], operator.add]
    symbol_confidence: float
    
    # OCR results
    extracted_labels: Annotated[Dict[str, str], operator.or_]
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

class Stream(BaseModel):
    id: str
    source: str = Field(..., description="Source equipment ID")
    target: str = Field(..., description="Target equipment ID")
    type: str = "stream"
    label: str | None = None
    confidence: float = 1.0

def build_graph(self, symbols: List[Dict], labels: Dict[str, str], connections: List[Dict]) -> Dict:
    equipment_list = []
    stream_list = []

    # Create equipment nodes first
    for idx, symbol in enumerate(symbols):
        equipment = Equipment(
            id=f"equipment_{idx}",
            type=symbol.get("type", "unknown"),
            label=labels.get(f"equipment_{idx}", ""),
            confidence=symbol.get("confidence", 1.0)
        )
        equipment_list.append(equipment)

    # Create streams with validation
    for idx, conn in enumerate(connections):
        # Validate source and target exist
        source = conn.get("source")
        target = conn.get("target")
        
        if not source or not target:
            logger.warning(f"Skipping stream {idx} due to missing source or target")
            continue
            
        try:
            stream = Stream(
                id=f"stream_{idx}",
                source=str(source),  # Ensure string type
                target=str(target),  # Ensure string type
                label=labels.get(f"stream_{idx}", ""),
                confidence=conn.get("confidence", 1.0)
            )
            stream_list.append(stream)
        except Exception as e:
            logger.error(f"Failed to create stream {idx}: {str(e)}")
            continue

    return {
        "equipment": [eq.model_dump() for eq in equipment_list],
        "streams": [stream.model_dump() for stream in stream_list]
    }

def graph_builder_node(self, state: PFDProcessingState) -> PFDProcessingState:
    # Validate required state fields
    if not state.get("detected_symbols"):
        raise ValueError("No detected symbols in state")
    if not state.get("extracted_labels"):
        raise ValueError("No extracted labels in state")
    if not state.get("connections"):
        raise ValueError("No connections in state")

    try:
        result = self.graph_builder.build_graph(
            symbols=state["detected_symbols"],
            labels=state["extracted_labels"],
            connections=state["connections"]
        )
        state["graph"] = result
        return state
    except Exception as e:
        logger.error(f"Graph building failed: {str(e)}")
        raise