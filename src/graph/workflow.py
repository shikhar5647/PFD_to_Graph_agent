# src/graph/workflow.py
from langgraph.graph import StateGraph, END
from typing import Dict
from .state import PFDProcessingState
from ..agents.vision_agent import VisionAgent
from ..agents.ocr_agent import OCRAgent
from ..agents.topology_agent import TopologyAgent
from ..agents.graph_builder_agent import GraphBuilderAgent
from ..agents.validator_agent import ValidatorAgent

class PFDWorkflow:
    """Main LangGraph workflow for PFD to Graph conversion using Gemini"""
    
    def __init__(self, llm_config: Dict):
        self.vision_agent = VisionAgent(llm_config)
        self.ocr_agent = OCRAgent(llm_config)
        self.topology_agent = TopologyAgent(llm_config)
        self.graph_builder = GraphBuilderAgent(llm_config)
        self.validator = ValidatorAgent(llm_config)
        
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(PFDProcessingState)
        
        # Add nodes
        workflow.add_node("vision_analysis", self.vision_node)
        workflow.add_node("ocr_extraction", self.ocr_node)
        workflow.add_node("topology_detection", self.topology_node)
        workflow.add_node("graph_construction", self.graph_builder_node)
        workflow.add_node("validation", self.validation_node)
        workflow.add_node("refinement", self.refinement_node)
        
        # Define edges
        workflow.set_entry_point("vision_analysis")
        workflow.add_edge("vision_analysis", "ocr_extraction")
        workflow.add_edge("ocr_extraction", "topology_detection")
        workflow.add_edge("topology_detection", "graph_construction")
        workflow.add_edge("graph_construction", "validation")
        
        # Conditional edge for validation
        workflow.add_conditional_edges(
            "validation",
            self.should_refine,
            {
                "refine": "refinement",
                "end": END
            }
        )
        workflow.add_edge("refinement", "graph_construction")
        
        return workflow.compile()
    
    def vision_node(self, state: PFDProcessingState) -> PFDProcessingState:
        """Vision analysis node - detect equipment symbols"""
        print("🔍 Vision Analysis Stage...")
        result = self.vision_agent.analyze(state["image"])
        
        return {
            **state,
            "detected_symbols": result["symbols"],
            "symbol_confidence": result["confidence"],
            "processing_stage": "vision_complete"
        }
    
    def ocr_node(self, state: PFDProcessingState) -> PFDProcessingState:
        """OCR node - extract text labels"""
        print("📝 OCR Extraction Stage...")
        result = self.ocr_agent.extract(
            state["image"],
            state["detected_symbols"]
        )
        
        return {
            **state,
            "extracted_labels": result["labels"],
            "text_regions": result["regions"],
            "processing_stage": "ocr_complete"
        }
    
    def topology_node(self, state: PFDProcessingState) -> PFDProcessingState:
        """Topology detection node - find connections"""
        print("🔗 Topology Detection Stage...")
        result = self.topology_agent.detect_connections(
            state["image"],
            state["detected_symbols"]
        )
        
        return {
            **state,
            "connections": result["connections"],
            "connection_confidence": result["confidence"],
            "processing_stage": "topology_complete"
        }
    
    def graph_builder_node(self, state: PFDProcessingState) -> PFDProcessingState:
        """Graph construction node - build NetworkX graph"""
        print("🏗️ Graph Construction Stage...")
        
        try:
            result = self.graph_builder.build_graph(
                symbols=state["detected_symbols"],
                labels=state["extracted_labels"],
                connections=state["connections"]
            )
            
            return {
                **state,
                "equipment_nodes": result["nodes"],
                "stream_edges": result["edges"],
                "graph_data": result["graph_dict"],
                "networkx_graph": result["networkx"],
                "processing_stage": "graph_complete"
            }
        except Exception as e:
            print(f"❌ Error in graph construction: {e}")
            import traceback
            traceback.print_exc()
            
            # Return state with error
            return {
                **state,
                "equipment_nodes": [],
                "stream_edges": [],
                "graph_data": None,
                "networkx_graph": None,
                "processing_stage": "graph_error",
                "errors": state.get("errors", []) + [f"Graph construction error: {str(e)}"]
            }
    
    def validation_node(self, state: PFDProcessingState) -> PFDProcessingState:
        """Validation node - validate graph structure"""
        print("✅ Validation Stage...")
        result = self.validator.validate(
            state["networkx_graph"],
            state["graph_data"]
        )
        
        return {
            **state,
            "validation_passed": result["passed"],
            "validation_errors": result["errors"],
            "warnings": result["warnings"],
            "processing_stage": "validation_complete"
        }
    
    def refinement_node(self, state: PFDProcessingState) -> PFDProcessingState:
        """Refinement node - fix validation errors"""
        print("🔧 Refinement Stage...")
        # Use LLM to analyze errors and suggest fixes
        result = self.validator.refine(
            state["networkx_graph"],
            state["validation_errors"]
        )
        
        return {
            **state,
            "equipment_nodes": result["refined_nodes"],
            "stream_edges": result["refined_edges"],
            "processing_stage": "refinement_complete"
        }
    
    def should_refine(self, state: PFDProcessingState) -> str:
        """Decide whether to refine the graph or finish"""
        if state["validation_passed"]:
            return "end"
        
        # Limit refinement attempts
        refinement_count = state.get("refinement_count", 0)
        if refinement_count >= 2:
            return "end"
        
        return "refine"
    
    def process(self, image_path: str, image) -> PFDProcessingState:
        """Process a PFD image through the workflow"""
        from PIL import Image
        
        # Ensure image is PIL Image
        if not isinstance(image, Image.Image):
            image = Image.open(image_path)
        
        initial_state = {
            "image": image,
            "image_path": image_path,
            "detected_symbols": [],
            "symbol_confidence": 0.0,
            "extracted_labels": {},
            "text_regions": [],
            "connections": [],
            "connection_confidence": 0.0,
            "equipment_nodes": [],
            "stream_edges": [],
            "graph_data": None,
            "networkx_graph": None,
            "validation_passed": False,
            "validation_errors": [],
            "processing_stage": "initialized",
            "errors": [],
            "warnings": []
        }
        
        # Run the workflow
        print("🚀 Starting PFD Processing Workflow...")
        final_state = self.workflow.invoke(initial_state)
        print("✅ Workflow Complete!")
        
        return final_state