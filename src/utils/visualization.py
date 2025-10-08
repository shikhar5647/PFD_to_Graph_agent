import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Tuple, Optional
import numpy as np
from pyvis.network import Network
import plotly.graph_objects as go

class GraphVisualizer:
    """Utilities for visualizing process flow graphs"""
    
    def __init__(self):
        self.color_map = {
            'reactor': '#FF6B6B',
            'heat_exchanger': '#4ECDC4',
            'distillation_column': '#45B7D1',
            'mixer': '#FFA07A',
            'splitter': '#98D8C8',
            'pump': '#6C5CE7',
            'compressor': '#A29BFE',
            'valve': '#74B9FF',
            'tank': '#81C784',
            'separator': '#FFB74D',
            'raw_material': '#E8F5E9',
            'product': '#FFF9C4',
            'utility': '#F5F5F5',
            'unknown': '#BDBDBD'
        }
        
        self.shape_map = {
            'reactor': 'o',
            'heat_exchanger': 's',
            'distillation_column': '^',
            'mixer': 'D',
            'splitter': 'v',
            'pump': 'p',
            'valve': 'h',
            'tank': '8',
            'raw_material': '>',
            'product': '<'
        }
    
    def get_node_color(self, equipment_type: str) -> str:
        """Get color for equipment type"""
        return self.color_map.get(equipment_type, self.color_map['unknown'])
    
    def get_node_shape(self, equipment_type: str) -> str:
        """Get shape for equipment type"""
        return self.shape_map.get(equipment_type, 'o')
    
    def get_layout(self, G: nx.DiGraph, layout_type: str = 'hierarchical') -> Dict:
        """
        Get node positions using various layout algorithms
        
        Args:
            G: NetworkX graph
            layout_type: Layout algorithm to use
        
        Returns:
            Dictionary of node positions
        """
        if layout_type == 'hierarchical':
            return self._hierarchical_layout(G)
        elif layout_type == 'spring':
            return nx.spring_layout(G, k=2, iterations=50)
        elif layout_type == 'circular':
            return nx.circular_layout(G)
        elif layout_type == 'kamada_kawai':
            return nx.kamada_kawai_layout(G)
        elif layout_type == 'planar':
            try:
                return nx.planar_layout(G)
            except:
                return nx.spring_layout(G)
        else:
            return nx.spring_layout(G)
    
    def _hierarchical_layout(self, G: nx.DiGraph) -> Dict:
        """
        Create hierarchical layout based on process flow direction
        """
        # Find source nodes (raw materials)
        source_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
        
        if not source_nodes:
            # If no pure sources, find nodes with lowest in-degree
            source_nodes = sorted(G.nodes(), key=lambda n: G.in_degree(n))[:1]
        
        # Calculate levels using BFS
        levels = {}
        visited = set()
        current_level = source_nodes
        level = 0
        
        while current_level:
            next_level = []
            for node in current_level:
                if node not in visited:
                    levels[node] = level
                    visited.add(node)
                    next_level.extend([n for n in G.successors(node) if n not in visited])
            current_level = next_level
            level += 1
        
        # Handle unvisited nodes
        for node in G.nodes():
            if node not in levels:
                levels[node] = level
        
        # Calculate positions
        pos = {}
        level_counts = {}
        level_indices = {}
        
        # Count nodes per level
        for node, lvl in levels.items():
            level_counts[lvl] = level_counts.get(lvl, 0) + 1
            level_indices[lvl] = level_indices.get(lvl, 0)
        
        # Assign positions
        for node, lvl in levels.items():
            x = lvl
            y = level_indices[lvl] - (level_counts[lvl] - 1) / 2
            pos[node] = (x, y)
            level_indices[lvl] += 1
        
        return pos
    
    def draw_static_graph(
        self,
        G: nx.DiGraph,
        pos: Optional[Dict] = None,
        layout: str = 'hierarchical',
        show_labels: bool = True,
        show_edge_labels: bool = False,
        figsize: Tuple[int, int] = (14, 10),
        title: str = "Process Flow Diagram"
    ) -> plt.Figure:
        """
        Draw static matplotlib graph
        
        Args:
            G: NetworkX graph
            pos: Node positions (if None, will be calculated)
            layout: Layout type to use
            show_labels: Show node labels
            show_edge_labels: Show edge labels
            figsize: Figure size
            title: Graph title
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        
        if pos is None:
            pos = self.get_layout(G, layout)
        
        # Group nodes by type for legend
        node_types = {}
        for node, data in G.nodes(data=True):
            eq_type = data.get('type', 'unknown')
            if eq_type not in node_types:
                node_types[eq_type] = []
            node_types[eq_type].append(node)
        
        # Draw nodes by type
        for eq_type, nodes in node_types.items():
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodes,
                node_color=self.get_node_color(eq_type),
                node_size=1200,
                alpha=0.9,
                node_shape=self.get_node_shape(eq_type),
                ax=ax,
                label=eq_type.replace('_', ' ').title()
            )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            edge_color='#34495e',
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            width=2,
            alpha=0.6,
            connectionstyle='arc3,rad=0.1',
            ax=ax
        )
        
        # Draw labels
        if show_labels:
            labels = {node: data.get('label', node) for node, data in G.nodes(data=True)}
            nx.draw_networkx_labels(
                G, pos,
                labels,
                font_size=9,
                font_weight='bold',
                font_color='black',
                ax=ax
            )
        
        if show_edge_labels:
            edge_labels = {
                (u, v): data.get('label', '')
                for u, v, data in G.edges(data=True)
                if data.get('label')
            }
            if edge_labels:
                nx.draw_networkx_edge_labels(
                    G, pos,
                    edge_labels,
                    font_size=7,
                    ax=ax
                )
        
        ax.axis('off')
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        
        # Add legend
        ax.legend(
            loc='upper left',
            bbox_to_anchor=(1, 1),
            frameon=True,
            fancybox=True,
            shadow=True
        )
        
        plt.tight_layout()
        return fig
    
    def create_interactive_pyvis(
        self,
        G: nx.DiGraph,
        height: str = "750px",
        width: str = "100%",
        notebook: bool = False
    ) -> Network:
        """
        Create interactive PyVis network
        
        Args:
            G: NetworkX graph
            height: Height of visualization
            width: Width of visualization
            notebook: Whether running in Jupyter notebook
        
        Returns:
            PyVis Network object
        """
        net = Network(
            height=height,
            width=width,
            directed=True,
            notebook=notebook,
            bgcolor='#ffffff',
            font_color='black'
        )
        
        # Add nodes
        for node, data in G.nodes(data=True):
            label = data.get('label', node)
            eq_type = data.get('type', 'unknown')
            color = self.get_node_color(eq_type)
            
            # Create hover title with details
            title = f"<b>{label}</b><br>"
            title += f"Type: {eq_type.replace('_', ' ').title()}<br>"
            
            if data.get('properties'):
                title += "<br><b>Properties:</b><br>"
                for k, v in data['properties'].items():
                    title += f"{k}: {v}<br>"
            
            net.add_node(
                node,
                label=label,
                title=title,
                color=color,
                size=25,
                font={'size': 14, 'face': 'arial'}
            )
        
        # Add edges
        for source, target, data in G.edges(data=True):
            edge_label = data.get('label', '')
            stream_type = data.get('stream_type', 'material')
            
            title = f"Stream: {stream_type}<br>"
            if data.get('properties'):
                for k, v in data['properties'].items():
                    title += f"{k}: {v}<br>"
            
            net.add_edge(
                source,
                target,
                label=edge_label,
                title=title,
                arrows='to',
                color={'color': '#34495e', 'opacity': 0.6},
                width=2
            )
        
        # Configure physics
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -30000,
              "centralGravity": 0.3,
              "springLength": 150,
              "springConstant": 0.04,
              "damping": 0.09,
              "avoidOverlap": 0.1
            },
            "minVelocity": 0.75,
            "solver": "barnesHut"
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true
          }
        }
        """)
        
        return net
    
    def create_plotly_graph(
        self,
        G: nx.DiGraph,
        layout: str = 'hierarchical',
        title: str = "Process Flow Diagram"
    ) -> go.Figure:
        """
        Create interactive Plotly graph
        
        Args:
            G: NetworkX graph
            layout: Layout algorithm
            title: Graph title
        
        Returns:
            Plotly figure
        """
        pos = self.get_layout(G, layout)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Add edge label
            edge_label = edge[2].get('label', '')
            if edge_label:
                edge_text.append(edge_label)
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node, data in G.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            label = data.get('label', node)
            eq_type = data.get('type', 'unknown')
            node_text.append(f"{label}<br>Type: {eq_type}")
            node_colors.append(self.get_node_color(eq_type))
            
            # Size based on degree
            degree = G.degree(node)
            node_sizes.append(20 + degree * 5)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[data.get('label', node) for node, data in G.nodes(data=True)],
            textposition='top center',
            hoverinfo='text',
            hovertext=node_text,
            marker=dict(
                showscale=False,
                color=node_colors,
                size=node_sizes,
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=title,
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
                height=700
            )
        )
        
        return fig
    
    def create_legend_figure(self) -> plt.Figure:
        """Create a standalone legend figure for equipment types"""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        
        patches = []
        for eq_type, color in self.color_map.items():
            if eq_type != 'unknown':
                label = eq_type.replace('_', ' ').title()
                patches.append(mpatches.Patch(color=color, label=label))
        
        ax.legend(
            handles=patches,
            loc='center',
            ncol=2,
            title='Equipment Types',
            title_fontsize=14,
            fontsize=11,
            frameon=True,
            fancybox=True,
            shadow=True
        )
        
        return fig
