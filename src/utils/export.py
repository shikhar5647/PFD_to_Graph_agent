import networkx as nx
import json
from typing import Dict, Any
import xml.etree.ElementTree as ET
from xml.dom import minidom
import csv
from io import StringIO

class GraphExporter:
    """Utilities for exporting graphs in various formats"""
    
    def __init__(self):
        pass
    
    def to_graphml(self, G: nx.DiGraph) -> str:
        """
        Export graph to GraphML format
        
        Args:
            G: NetworkX graph
        
        Returns:
            GraphML as string
        """
        # Create GraphML structure
        graphml = ET.Element('graphml', {
            'xmlns': 'http://graphml.graphdrawing.org/xmlns',
            'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
            'xsi:schemaLocation': 'http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd'
        })
        
        # Define keys for node and edge attributes
        node_keys = set()
        edge_keys = set()
        
        for _, data in G.nodes(data=True):
            node_keys.update(data.keys())
        
        for _, _, data in G.edges(data=True):
            edge_keys.update(data.keys())
        
        # Add key definitions
        for i, key in enumerate(node_keys):
            ET.SubElement(graphml, 'key', {
                'id': f'n{i}',
                'for': 'node',
                'attr.name': key,
                'attr.type': 'string'
            })
        
        for i, key in enumerate(edge_keys):
            ET.SubElement(graphml, 'key', {
                'id': f'e{i}',
                'for': 'edge',
                'attr.name': key,
                'attr.type': 'string'
            })
        
        # Create graph element
        graph = ET.SubElement(graphml, 'graph', {
            'id': 'PFD',
            'edgedefault': 'directed'
        })
        
        # Add nodes
        node_key_map = {key: f'n{i}' for i, key in enumerate(node_keys)}
        for node, data in G.nodes(data=True):
            node_elem = ET.SubElement(graph, 'node', {'id': str(node)})
            for key, value in data.items():
                data_elem = ET.SubElement(node_elem, 'data', {'key': node_key_map[key]})
                data_elem.text = str(value)
        
        # Add edges
        edge_key_map = {key: f'e{i}' for i, key in enumerate(edge_keys)}
        for i, (source, target, data) in enumerate(G.edges(data=True)):
            edge_elem = ET.SubElement(graph, 'edge', {
                'id': f'e{i}',
                'source': str(source),
                'target': str(target)
            })
            for key, value in data.items():
                data_elem = ET.SubElement(edge_elem, 'data', {'key': edge_key_map[key]})
                data_elem.text = str(value)
        
        # Convert to pretty XML string
        xml_str = minidom.parseString(ET.tostring(graphml)).toprettyxml(indent="  ")
        return xml_str
    
    def to_json(self, G: nx.DiGraph, indent: int = 2) -> str:
        """
        Export graph to JSON format
        
        Args:
            G: NetworkX graph
            indent: JSON indentation
        
        Returns:
            JSON as string
        """
        data = {
            'directed': True,
            'multigraph': False,
            'graph': dict(G.graph),
            'nodes': [
                {'id': node, **data}
                for node, data in G.nodes(data=True)
            ],
            'edges': [
                {'source': source, 'target': target, **data}
                for source, target, data in G.edges(data=True)
            ]
        }
        
        return json.dumps(data, indent=indent, default=str)
    
    def to_gml(self, G: nx.DiGraph) -> str:
        """
        Export graph to GML format
        
        Args:
            G: NetworkX graph
        
        Returns:
            GML as string
        """
        lines = ['graph [', '  directed 1']
        
        # Add graph attributes
        for key, value in G.graph.items():
            lines.append(f'  {key} "{value}"')
        
        # Add nodes
        for node, data in G.nodes(data=True):
            lines.append('  node [')
            lines.append(f'    id {node}')
            for key, value in data.items():
                if isinstance(value, str):
                    lines.append(f'    {key} "{value}"')
                else:
                    lines.append(f'    {key} {value}')
            lines.append('  ]')
        
        # Add edges
        for source, target, data in G.edges(data=True):
            lines.append('  edge [')
            lines.append(f'    source {source}')
            lines.append(f'    target {target}')
            for key, value in data.items():
                if isinstance(value, str):
                    lines.append(f'    {key} "{value}"')
                else:
                    lines.append(f'    {key} {value}')
            lines.append('  ]')
        
        lines.append(']')
        
        return '\n'.join(lines)
    
    def to_csv(self, G: nx.DiGraph) -> str:
        """
        Export graph to CSV edge list format
        
        Args:
            G: NetworkX graph
        
        Returns:
            CSV as string
        """
        output = StringIO()
        writer = csv.writer(output)
        
        # Get all edge attribute keys
        edge_keys = set()
        for _, _, data in G.edges(data=True):
            edge_keys.update(data.keys())
        
        edge_keys = sorted(edge_keys)
        
        # Write header
        header = ['source', 'target'] + edge_keys
        writer.writerow(header)
        
        # Write edges
        for source, target, data in G.edges(data=True):
            row = [source, target]
            for key in edge_keys:
                row.append(data.get(key, ''))
            writer.writerow(row)
        
        return output.getvalue()
    
    def to_adjacency_matrix(self, G: nx.DiGraph) -> str:
        """
        Export graph as adjacency matrix CSV
        
        Args:
            G: NetworkX graph
        
        Returns:
            CSV adjacency matrix as string
        """
        nodes = list(G.nodes())
        matrix = nx.to_numpy_array(G, nodelist=nodes, dtype=int)
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([''] + nodes)
        
        # Write matrix
        for i, node in enumerate(nodes):
            row = [node] + matrix[i].tolist()
            writer.writerow(row)
        
        return output.getvalue()
    
    def to_dot(self, G: nx.DiGraph) -> str:
        """
        Export graph to DOT format (Graphviz)
        
        Args:
            G: NetworkX graph
        
        Returns:
            DOT format as string
        """
        lines = ['digraph PFD {']
        lines.append('  rankdir=LR;')
        lines.append('  node [shape=box, style=filled];')
        
        # Add nodes
        for node, data in G.nodes(data=True):
            label = data.get('label', node)
            eq_type = data.get('type', 'unknown')
            
            attrs = [
                f'label="{label}"',
                f'type="{eq_type}"'
            ]
            
            lines.append(f'  "{node}" [{", ".join(attrs)}];')
        
        # Add edges
        for source, target, data in G.edges(data=True):
            label = data.get('label', '')
            
            if label:
                lines.append(f'  "{source}" -> "{target}" [label="{label}"];')
            else:
                lines.append(f'  "{source}" -> "{target}";')
        
        lines.append('}')
        
        return '\n'.join(lines)
    
    def to_sfiles(self, G: nx.DiGraph) -> str:
        """
        Export graph to SFILES 2.0 format (custom format for process systems)
        
        Args:
            G: NetworkX graph
        
        Returns:
            SFILES format as string
        """
        lines = ['# SFILES 2.0 Format', '# Process Flow Diagram', '']
        
        # Equipment section
        lines.append('[EQUIPMENT]')
        for node, data in G.nodes(data=True):
            label = data.get('label', node)
            eq_type = data.get('type', 'unknown')
            lines.append(f'{node}\t{label}\t{eq_type}')
        
        lines.append('')
        
        # Streams section
        lines.append('[STREAMS]')
        for i, (source, target, data) in enumerate(G.edges(data=True)):
            stream_id = data.get('stream_id', f'S{i+1}')
            stream_type = data.get('stream_type', 'material')
            lines.append(f'{stream_id}\t{source}\t{target}\t{stream_type}')
        
        lines.append('')
        
        # Topology section
        lines.append('[TOPOLOGY]')
        lines.append(f'NODES={G.number_of_nodes()}')
        lines.append(f'EDGES={G.number_of_edges()}')
        lines.append(f'CONNECTED={nx.is_weakly_connected(G)}')
        
        return '\n'.join(lines)
