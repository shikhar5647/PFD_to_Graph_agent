import streamlit as st
import sys
from pathlib import Path
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import tempfile
import os

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.graph.workflow import PFDWorkflow
from src.utils.visualization import GraphVisualizer
from src.utils.export import GraphExporter

# Page configuration
st.set_page_config(
    page_title="PFD to Graph Converter",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
    }
    </style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'processed_graph' not in st.session_state:
        st.session_state.processed_graph = None
    if 'processing_state' not in st.session_state:
        st.session_state.processing_state = None
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None

def main():
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🏭 PFD to Graph Converter</h1>', unsafe_allow_html=True)
    st.markdown("Convert Process Flow Diagrams to Graph-based Notation using AI Agents")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # LLM Settings
        st.subheader("LLM Settings")
        api_key = st.text_input(
            "Gemini API key",
            type="password",
            help="Enter your Gemini API key"
        )
        
        model = st.selectbox(
            "Model",
            ["gemini-2.5-pro", "gemini-2.5-flash"],
            index=0
        )
        
        # Processing Options
        st.subheader("Processing Options")
        use_cv_backup = st.checkbox("Use CV Backup Detection", value=True)
        enable_validation = st.checkbox("Enable Graph Validation", value=True)
        auto_refinement = st.checkbox("Auto Refinement", value=True)
        
        # Visualization Options
        st.subheader("Visualization")
        viz_layout = st.selectbox(
            "Graph Layout",
            ["hierarchical", "spring", "circular", "kamada_kawai"],
            index=0
        )
        
        show_labels = st.checkbox("Show Labels", value=True)
        show_attributes = st.checkbox("Show Attributes", value=False)
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload & Process",
        "📊 Graph View",
        "📋 Details",
        "💾 Export"
    ])
    
    with tab1:
        upload_and_process_tab(api_key, model, use_cv_backup, enable_validation)
    
    with tab2:
        graph_view_tab(viz_layout, show_labels, show_attributes)
    
    with tab3:
        details_tab()
    
    with tab4:
        export_tab()

def upload_and_process_tab(api_key, model, use_cv_backup, enable_validation):
    """Upload and process PFD images"""
    st.header("Upload PFD Image")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a PFD image",
            type=['png', 'jpg', 'jpeg', 'pdf'],
            help="Upload a Process Flow Diagram image"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.session_state.uploaded_image = image
            st.image(image, caption="Uploaded PFD", use_column_width=True)
    
    with col2:
        if st.session_state.uploaded_image is not None:
            st.subheader("Image Information")
            st.write(f"**Size:** {st.session_state.uploaded_image.size}")
            st.write(f"**Format:** {st.session_state.uploaded_image.format}")
            st.write(f"**Mode:** {st.session_state.uploaded_image.mode}")
            
            # Process button
            if st.button("🚀 Process PFD", type="primary", use_container_width=True):
                if not api_key:
                    st.error("Please enter your Anthropic API key in the sidebar")
                    return
                
                process_pfd(
                    st.session_state.uploaded_image,
                    api_key,
                    model,
                    use_cv_backup,
                    enable_validation
                )

def process_pfd(image, api_key, model, use_cv_backup, enable_validation):
    """Process PFD image through the workflow"""
    with st.spinner("Processing PFD... This may take a minute."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Initialize workflow
            llm_config = {
                "model": model,
                "api_key": api_key
            }
            
            workflow = PFDWorkflow(llm_config)
            
            # Save temp image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                image.save(tmp.name)
                tmp_path = tmp.name
            
            # Process through workflow
            status_text.text("🔍 Stage 1/5: Vision Analysis...")
            progress_bar.progress(20)
            
            status_text.text("📝 Stage 2/5: OCR Extraction...")
            progress_bar.progress(40)
            
            status_text.text("🔗 Stage 3/5: Topology Detection...")
            progress_bar.progress(60)
            
            status_text.text("🏗️ Stage 4/5: Graph Construction...")
            progress_bar.progress(80)
            
            final_state = workflow.process(tmp_path, image)
            
            status_text.text("✅ Stage 5/5: Validation...")
            progress_bar.progress(100)
            
            # Store results
            st.session_state.processing_state = final_state
            st.session_state.processed_graph = final_state.get('networkx_graph')
            
            # Clean up
            os.unlink(tmp_path)
            
            # Display results
            if final_state.get('validation_passed'):
                st.success("✅ PFD processed successfully!")
            else:
                st.warning("⚠️ Processing complete with warnings")
            
            # Show summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nodes Detected", len(final_state.get('equipment_nodes', [])))
            with col2:
                st.metric("Edges Detected", len(final_state.get('stream_edges', [])))
            with col3:
                confidence = final_state.get('symbol_confidence', 0) * 100
                st.metric("Confidence", f"{confidence:.1f}%")
            
        except Exception as e:
            st.error(f"Error processing PFD: {str(e)}")
            st.exception(e)

def graph_view_tab(layout, show_labels, show_attributes):
    """Display graph visualization"""
    st.header("Graph Visualization")
    
    if st.session_state.processed_graph is None:
        st.info("👆 Please upload and process a PFD image first")
        return
    
    G = st.session_state.processed_graph
    
    # Create visualization
    visualizer = GraphVisualizer()
    
    viz_type = st.radio(
        "Visualization Type",
        ["Interactive", "Static", "3D (if available)"],
        horizontal=True
    )
    
    if viz_type == "Interactive":
        # Use PyVis for interactive visualization
        net = Network(height="600px", width="100%", directed=True)
        
        # Add nodes
        for node, data in G.nodes(data=True):
            label = data.get('label', node) if show_labels else ''
            title = f"{node}\nType: {data.get('type', 'unknown')}"
            
            if show_attributes:
                for key, value in data.items():
                    if key not in ['label', 'type']:
                        title += f"\n{key}: {value}"
            
            color = visualizer.get_node_color(data.get('type'))
            net.add_node(node, label=label, title=title, color=color)
        
        # Add edges
        for source, target, data in G.edges(data=True):
            label = data.get('label', '') if show_labels else ''
            title = f"Stream: {data.get('stream_type', 'material')}"
            net.add_edge(source, target, label=label, title=title, arrows='to')
        
        # Set physics
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "hierarchicalRepulsion": {
              "centralGravity": 0.3,
              "springLength": 150,
              "springConstant": 0.01,
              "nodeDistance": 200
            },
            "solver": "hierarchicalRepulsion"
          }
        }
        """)
        
        # Save and display
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
            net.save_graph(tmp.name)
            with open(tmp.name, 'r') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600, scrolling=True)
            os.unlink(tmp.name)
    
    elif viz_type == "Static":
        # Use matplotlib for static visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        pos = visualizer.get_layout(G, layout)
        
        # Draw nodes
        node_colors = [
            visualizer.get_node_color(data.get('type'))
            for node, data in G.nodes(data=True)
        ]
        
        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors,
            node_size=1000, alpha=0.9, ax=ax
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos, edge_color='gray',
            arrows=True, arrowsize=20, ax=ax,
            connectionstyle='arc3,rad=0.1'
        )
        
        # Draw labels
        if show_labels:
            labels = {
                node: data.get('label', node)
                for node, data in G.nodes(data=True)
            }
            nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)
        
        ax.axis('off')
        ax.set_title("Process Flow Graph", fontsize=16, fontweight='bold')
        
        st.pyplot(fig)
    
    # Graph statistics
    with st.expander("📈 Graph Statistics"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Nodes", G.number_of_nodes())
        with col2:
            st.metric("Total Edges", G.number_of_edges())
        with col3:
            st.metric("Avg Degree", f"{sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
        with col4:
            st.metric("Is Connected", "Yes" if nx.is_weakly_connected(G) else "No")

def details_tab():
    """Display detailed information"""
    st.header("Processing Details")
    
    if st.session_state.processing_state is None:
        st.info("👆 Please upload and process a PFD image first")
        return
    
    state = st.session_state.processing_state
    
    # Processing stages
    st.subheader("Processing Stages")
    stages = [
        ("Vision Analysis", "detected_symbols", "symbol_confidence"),
        ("OCR Extraction", "extracted_labels", None),
        ("Topology Detection", "connections", "connection_confidence"),
        ("Graph Construction", "equipment_nodes", None),
        ("Validation", "validation_errors", None)
    ]
    
    for stage_name, key, confidence_key in stages:
        with st.expander(f"📍 {stage_name}"):
            data = state.get(key, [])
            
            if isinstance(data, list):
                st.write(f"Found {len(data)} items")
                if data:
                    st.json(data[:3])  # Show first 3 items
            elif isinstance(data, dict):
                st.write(f"Found {len(data)} items")
                st.json(dict(list(data.items())[:5]))  # Show first 5 items
            
            if confidence_key:
                confidence = state.get(confidence_key, 0) * 100
                st.metric("Confidence", f"{confidence:.1f}%")
    
    # Validation results
    if state.get('validation_errors'):
        st.subheader("⚠️ Validation Issues")
        for error in state['validation_errors']:
            st.error(error)
    
    if state.get('warnings'):
        st.subheader("⚠️ Warnings")
        for warning in state['warnings']:
            st.warning(warning)

def export_tab():
    """Export graph in various formats"""
    st.header("Export Graph")
    
    if st.session_state.processed_graph is None:
        st.info("👆 Please upload and process a PFD image first")
        return
    
    G = st.session_state.processed_graph
    exporter = GraphExporter()
    
    st.subheader("Export Formats")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # GraphML export
        if st.button("📥 Download GraphML"):
            graphml_data = exporter.to_graphml(G)
            st.download_button(
                label="Download GraphML",
                data=graphml_data,
                file_name="pfd_graph.graphml",
                mime="application/xml"
            )
        
        # JSON export
        if st.button("📥 Download JSON"):
            json_data = exporter.to_json(G)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name="pfd_graph.json",
                mime="application/json"
            )
    
    with col2:
        # GML export
        if st.button("📥 Download GML"):
            gml_data = exporter.to_gml(G)
            st.download_button(
                label="Download GML",
                data=gml_data,
                file_name="pfd_graph.gml",
                mime="text/plain"
            )
        
        # CSV export (edge list)
        if st.button("📥 Download CSV"):
            csv_data = exporter.to_csv(G)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="pfd_graph.csv",
                mime="text/csv"
            )
    
    # Preview
    st.subheader("Preview")
    
    format_preview = st.selectbox(
        "Format",
        ["GraphML", "JSON", "Edge List"]
    )
    
    if format_preview == "GraphML":
        st.code(exporter.to_graphml(G), language="xml")
    elif format_preview == "JSON":
        st.code(exporter.to_json(G), language="json")
    elif format_preview == "Edge List":
        st.code(exporter.to_csv(G), language="text")


if __name__ == "__main__":
    main()