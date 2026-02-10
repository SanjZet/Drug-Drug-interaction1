"""
Drug Interaction Network Visualization

This module provides visualization tools for the drug interaction network:
- Network graphs with different layouts
- Degree distribution analysis
- Interactive plotly visualizations
- Subgraph exploration
- Community detection
"""

import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from torch_geometric.data import Data

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly not available. Install with: pip install plotly")


class DrugNetworkVisualizer:
    """Visualize drug interaction networks"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.graph = None
        self.pos = None
        self.drug_names = {}
        
    def load_data_from_pytorch(self):
        """Load drug network from PyTorch graph file (faster for large datasets)"""
        print("Loading data from PyTorch graph...")
        
        graph_path = self.data_dir / 'drug_graph.pt'
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_path}")
        
        # Load PyTorch Geometric data
        print("  📁 Loading drug_graph.pt...")
        saved_data = torch.load(graph_path, weights_only=False)
        
        # Extract graph data
        graph_data = saved_data['graph_data']
        idx_to_drug = saved_data['idx_to_drug']
        
        # Get edge index and number of nodes
        edge_index = graph_data.edge_index
        num_nodes = graph_data.x.shape[0]
        num_edges = edge_index.shape[1]
        
        print(f"     ✓ Graph loaded ({num_nodes} nodes, {num_edges} edges)")
        
        # Create NetworkX graph
        print("  🔨 Converting to NetworkX...")
        self.graph = nx.Graph()
        
        # Add nodes
        for i in range(num_nodes):
            node_name = idx_to_drug.get(i, f"Drug_{i}")
            self.drug_names[i] = node_name
            self.graph.add_node(i, name=node_name)
        
        # Add edges
        edge_numpy = edge_index.cpu().numpy()
        for i in range(edge_numpy.shape[1]):
            src, dst = int(edge_numpy[0, i]), int(edge_numpy[1, i])
            if src < dst:  # Add each edge only once (since graph is undirected)
                self.graph.add_edge(src, dst)
        
        print(f"\n✅ Graph built successfully!")
        print(f"   Nodes: {self.graph.number_of_nodes()} drugs")
        print(f"   Edges: {self.graph.number_of_edges()} interactions\n")
        
        return self.graph
    
    def load_data(self):
        """Load drug and interaction data"""
        print("Loading data...")
        
        # Load drugs
        print("  📁 Loading drugs.csv...")
        drugs_df = pd.read_csv(self.data_dir / 'drugs.csv')
        print(f"     ✓ {len(drugs_df)} drugs loaded")
        
        # Load interactions with chunking for large files
        print("  📁 Loading interactions.csv...")
        interactions_df = pd.read_csv(
            self.data_dir / 'interactions.csv',
            low_memory=False
        )
        print(f"     ✓ {len(interactions_df)} interactions loaded")
        
        # Create NetworkX graph
        print("  🔨 Building network graph...")
        self.graph = nx.Graph()
        
        # Add nodes
        for _, drug in drugs_df.iterrows():
            self.graph.add_node(
                drug['drug_id'],
                name=drug['name'],
                type=drug.get('type', 'unknown')
            )
        
        # Add edges (with progress indicator for large datasets)
        edges_added = 0
        for _, interaction in interactions_df.iterrows():
            if interaction['drug_1'] in self.graph.nodes and \
               interaction['drug_2'] in self.graph.nodes:
                self.graph.add_edge(
                    interaction['drug_1'],
                    interaction['drug_2'],
                    description=interaction.get('description', '')
                )
                edges_added += 1
        
        print(f"\n✅ Graph built successfully!")
        print(f"   Nodes: {self.graph.number_of_nodes()} drugs")
        print(f"   Edges: {self.graph.number_of_edges()} interactions\n")
        
        return self.graph
    
    def plot_network(self, layout='spring', figsize=(15, 12), 
                     node_size_by_degree=True, save_path=None, max_nodes=1000):
        """
        Plot the drug interaction network
        
        Args:
            layout: 'spring', 'circular', 'kamada_kawai', 'spectral'
            figsize: Figure size
            node_size_by_degree: Size nodes by their degree
            save_path: Path to save figure
            max_nodes: Maximum nodes to visualize (sample if graph is larger)
        """
        if self.graph is None:
            self.load_data_from_pytorch()
        
        print(f"Creating {layout} layout visualization...")
        
        # Sample graph if too large
        G = self.graph
        if G.number_of_nodes() > max_nodes:
            print(f"  ⚠️  Graph has {G.number_of_nodes()} nodes, sampling {max_nodes}...")
            # Sample high-degree nodes (most important)
            degrees = dict(G.degree())
            top_nodes = sorted(degrees.keys(), key=lambda n: degrees[n], reverse=True)[:max_nodes]
            G = G.subgraph(top_nodes).copy()
            print(f"  ✓ Sampled subgraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Create layout
        if layout == 'spring':
            self.pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        elif layout == 'circular':
            self.pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            self.pos = nx.kamada_kawai_layout(G)
        elif layout == 'spectral':
            self.pos = nx.spectral_layout(G)
        else:
            self.pos = nx.spring_layout(G)
        
        # Calculate node sizes
        if node_size_by_degree:
            degrees = dict(G.degree())
            node_sizes = [degrees[node] * 50 + 100 for node in G.nodes()]
        else:
            node_sizes = 300
        
        # Calculate node colors by degree
        degrees = dict(G.degree())
        node_colors = [degrees[node] for node in G.nodes()]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw network
        nx.draw_networkx_edges(
            G, self.pos,
            alpha=0.2,
            width=0.5,
            edge_color='gray',
            ax=ax
        )
        
        nodes = nx.draw_networkx_nodes(
            G, self.pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap='YlOrRd',
            alpha=0.8,
            ax=ax
        )
        
        # Add labels for high-degree nodes
        high_degree_nodes = {
            node: G.nodes[node]['name']
            for node in G.nodes()
            if degrees[node] > np.percentile(list(degrees.values()), 75)
        }
        
        nx.draw_networkx_labels(
            G, self.pos,
            labels=high_degree_nodes,
            font_size=8,
            font_weight='bold',
            ax=ax
        )
        
        # Add colorbar
        plt.colorbar(nodes, label='Node Degree (# of Interactions)', ax=ax)
        
        plt.title(f'Drug Interaction Network ({layout.title()} Layout)\n'
                  f'{G.number_of_nodes()} Drugs, '
                  f'{G.number_of_edges()} Interactions',
                  fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved to {save_path}")
        else:
            save_path = self.data_dir / f'network_{layout}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved to {save_path}")
        
        plt.close()
        
    def plot_degree_distribution(self, save_path=None):
        """Plot degree distribution analysis"""
        if self.graph is None:
            self.load_data()
        
        print("Analyzing degree distribution...")
        
        degrees = [d for n, d in self.graph.degree()]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Histogram
        axes[0, 0].hist(degrees, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 0].set_xlabel('Degree (# of Interactions)', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('Degree Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Box plot
        axes[0, 1].boxplot(degrees, vert=True)
        axes[0, 1].set_ylabel('Degree', fontsize=12)
        axes[0, 1].set_title('Degree Distribution (Box Plot)', fontsize=14, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Log-log plot (power law check)
        degree_counts = pd.Series(degrees).value_counts().sort_index()
        axes[1, 0].loglog(degree_counts.index, degree_counts.values, 'bo-', alpha=0.7)
        axes[1, 0].set_xlabel('Degree (log scale)', fontsize=12)
        axes[1, 0].set_ylabel('Frequency (log scale)', fontsize=12)
        axes[1, 0].set_title('Log-Log Degree Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].grid(alpha=0.3)
        
        # 4. Top drugs by degree
        top_drugs = sorted(self.graph.degree(), key=lambda x: x[1], reverse=True)[:15]
        drug_names = [self.graph.nodes[d[0]]['name'][:20] for d in top_drugs]
        drug_degrees = [d[1] for d in top_drugs]
        
        axes[1, 1].barh(range(len(drug_names)), drug_degrees, color='coral')
        axes[1, 1].set_yticks(range(len(drug_names)))
        axes[1, 1].set_yticklabels(drug_names, fontsize=9)
        axes[1, 1].set_xlabel('Degree', fontsize=12)
        axes[1, 1].set_title('Top 15 Most Connected Drugs', fontsize=14, fontweight='bold')
        axes[1, 1].invert_yaxis()
        axes[1, 1].grid(alpha=0.3, axis='x')
        
        plt.suptitle(f'Network Statistics\n'
                     f'Avg Degree: {np.mean(degrees):.2f} | '
                     f'Median: {np.median(degrees):.0f} | '
                     f'Max: {max(degrees)}',
                     fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        # Save
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved to {save_path}")
        else:
            save_path = self.data_dir / 'degree_distribution.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved to {save_path}")
        
        plt.close()
        
    def plot_network_statistics(self, save_path=None):
        """Plot comprehensive network statistics"""
        if self.graph is None:
            self.load_data_from_pytorch()
        
        print("Computing network statistics...")
        
        # Compute basic statistics
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        
        stats = {
            'Nodes': num_nodes,
            'Edges': num_edges,
            'Density': nx.density(self.graph),
        }
        
        # Skip expensive computations for very large graphs
        is_large_graph = num_nodes > 1000
        
        if not is_large_graph:
            print("  Computing clustering coefficient...")
            stats['Avg Clustering'] = nx.average_clustering(self.graph)
        else:
            print("  ⚠️  Skipping clustering (graph too large)")
            stats['Avg Clustering'] = None
        
        # Try to compute more stats (may fail for disconnected graphs)
        if not is_large_graph:
            try:
                stats['Avg Path Length'] = nx.average_shortest_path_length(self.graph)
            except:
                stats['Avg Path Length'] = None
            
            try:
                stats['Diameter'] = nx.diameter(self.graph)
            except:
                stats['Diameter'] = None
        else:
            stats['Avg Path Length'] = None
            stats['Diameter'] = None
        
        # Connected components
        print("  Computing connected components...")
        components = list(nx.connected_components(self.graph))
        stats['Connected Components'] = len(components)
        stats['Largest Component Size'] = len(max(components, key=len))
        
        # Degree statistics
        print("  Computing degree statistics...")
        degrees = [d for n, d in self.graph.degree()]
        stats['Avg Degree'] = np.mean(degrees)
        stats['Median Degree'] = np.median(degrees)
        stats['Max Degree'] = max(degrees)
        
        # Print statistics
        print("\n" + "="*60)
        print("NETWORK STATISTICS".center(60))
        print("="*60)
        for key, value in stats.items():
            if value is None:
                print(f"{key:.<40} N/A")
            elif isinstance(value, float):
                print(f"{key:.<40} {value:.4f}")
            else:
                print(f"{key:.<40} {value}")
        print("="*60 + "\n")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Create table
        table_data = []
        for key, value in stats.items():
            if value is None:
                table_data.append([key, 'N/A'])
            elif isinstance(value, float):
                table_data.append([key, f"{value:.4f}"])
            else:
                table_data.append([key, str(value)])
        
        table = ax.table(
            cellText=table_data,
            colLabels=['Metric', 'Value'],
            cellLoc='left',
            loc='center',
            colWidths=[0.6, 0.4]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 3)
        
        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            if i % 2 == 0:
                for j in range(2):
                    table[(i, j)].set_facecolor('#E7E6E6')
        
        plt.title('Drug Interaction Network Statistics', 
                  fontsize=16, fontweight='bold', pad=20)
        
        # Save
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved to {save_path}")
        else:
            save_path = self.data_dir / 'network_statistics.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved to {save_path}")
        
        plt.close()
        
        return stats
    
    def create_interactive_network(self, save_path=None):
        """Create interactive Plotly network visualization"""
        if not PLOTLY_AVAILABLE:
            print("❌ Plotly not installed. Run: pip install plotly")
            return
        
        if self.graph is None:
            self.load_data()
        
        print("Creating interactive network...")
        
        # Create layout
        pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        
        # Create edge traces
        edge_trace = go.Scatter(
            x=[], y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])
        
        # Create node traces
        node_trace = go.Scatter(
            x=[], y=[],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlOrRd',
                size=[],
                color=[],
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2, color='white')
            ),
            text=[]
        )
        
        for node in self.graph.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            
            # Node info
            degree = self.graph.degree(node)
            node_info = f"{self.graph.nodes[node]['name']}<br>"
            node_info += f"Interactions: {degree}"
            node_trace['text'] += tuple([node_info])
            node_trace['marker']['color'] += tuple([degree])
            node_trace['marker']['size'] += tuple([degree * 3 + 10])
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text='Interactive Drug Interaction Network',
                    x=0.5,
                    xanchor='center',
                    font=dict(size=20)
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
            )
        )
        
        # Save
        if save_path:
            fig.write_html(save_path)
            print(f"✅ Saved to {save_path}")
        else:
            save_path = self.data_dir / 'interactive_network.html'
            fig.write_html(save_path)
            print(f"✅ Saved to {save_path}")
            print(f"💡 Open {save_path} in your browser to interact!")


def main():
    """Run all visualizations"""
    print("\n" + "="*60)
    print("DRUG INTERACTION NETWORK VISUALIZATION".center(60))
    print("="*60 + "\n")
    
    viz = DrugNetworkVisualizer()
    
    try:
        # Load data from PyTorch graph (faster for large datasets)
        viz.load_data_from_pytorch()
        
        # 1. Network statistics
        print("\n[1/5] Computing network statistics...")
        viz.plot_network_statistics()
        
        # 2. Degree distribution
        print("\n[2/5] Plotting degree distribution...")
        viz.plot_degree_distribution()
        
        # 3. Spring layout
        print("\n[3/5] Creating spring layout network...")
        viz.plot_network(layout='spring')
        
        # 4. Circular layout
        print("\n[4/5] Creating circular layout network...")
        viz.plot_network(layout='circular')
        
        # 5. Interactive network
        print("\n[5/5] Creating interactive network...")
        viz.create_interactive_network()
        
        print("\n" + "="*60)
        print("✅ ALL VISUALIZATIONS COMPLETE!".center(60))
        print("="*60)
        print("\nGenerated files in 'data/' folder:")
        print("  📊 network_statistics.png")
        print("  📊 degree_distribution.png")
        print("  📊 network_spring.png")
        print("  📊 network_circular.png")
        print("  🌐 interactive_network.html")
        print("\n💡 Open the HTML file in your browser for interactive exploration!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
