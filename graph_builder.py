"""
Drug Interaction Project - Graph Builder
Converts parsed drug data into graph structure for GCN
"""

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json

class DrugGraphBuilder:
    """Build graph structure from drug interaction data"""
    
    def __init__(self, drugs_df, interactions_df):
        self.drugs_df = drugs_df
        self.interactions_df = interactions_df
        self.drug_to_idx = None
        self.idx_to_drug = None
        self.graph_data = None
        
    def build_graph(self, use_text_features=True, max_text_features=100):
        """
        Construct graph from drug data
        
        Args:
            use_text_features: Include text embeddings from descriptions
            max_text_features: Max dimensions for text features
        
        Returns:
            PyTorch Geometric Data object
        """
        print("Building drug interaction graph...")
        
        # Create drug ID mappings
        self._create_mappings()
        
        # Build edges (interactions)
        edge_index, edge_attributes = self._build_edges()
        
        # Build node features
        node_features = self._build_node_features(use_text_features, max_text_features)
        
        # Create PyTorch Geometric Data object
        self.graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attributes,
            num_nodes=len(self.drug_to_idx)
        )
        
        print(f"\n✅ Graph built successfully!")
        print(f"   Nodes (drugs): {self.graph_data.num_nodes}")
        print(f"   Edges (interactions): {self.graph_data.edge_index.shape[1] // 2}")
        print(f"   Node features: {self.graph_data.x.shape[1]}")
        
        return self.graph_data
    
    def _create_mappings(self):
        """Create drug ID to index mappings"""
        unique_drugs = self.drugs_df['drug_id'].unique()
        self.drug_to_idx = {drug_id: idx for idx, drug_id in enumerate(unique_drugs)}
        self.idx_to_drug = {idx: drug_id for drug_id, idx in self.drug_to_idx.items()}
        
        print(f"Created mappings for {len(self.drug_to_idx)} drugs")
    
    def _build_edges(self):
        """Build edge list and attributes from interactions"""
        edge_list = []
        edge_attrs = []
        
        print("Building edges from interactions...")
        for _, row in self.interactions_df.iterrows():
            drug1_id = row['drug_id_1']
            drug2_id = row['drug_id_2']
            
            # Skip if drug not in our drug list
            if drug1_id not in self.drug_to_idx or drug2_id not in self.drug_to_idx:
                continue
            
            drug1_idx = self.drug_to_idx[drug1_id]
            drug2_idx = self.drug_to_idx[drug2_id]
            
            # Add both directions (undirected graph)
            edge_list.append([drug1_idx, drug2_idx])
            edge_list.append([drug2_idx, drug1_idx])
            
            # Edge attributes (severity score)
            severity = self._classify_severity(row['description'])
            edge_attrs.extend([severity, severity])
        
        # Convert to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(1)
        
        return edge_index, edge_attr
    
    def _classify_severity(self, description):
        """
        Classify interaction severity from description
        Returns: 0.0-1.0 score (higher = more severe)
        """
        if pd.isna(description):
            return 0.5
        
        desc_lower = description.lower()
        
        # Critical keywords
        critical = ['contraindicated', 'avoid', 'do not', 'fatal', 'life-threatening', 
                   'severe', 'serious cardiovascular', 'black box']
        if any(word in desc_lower for word in critical):
            return 1.0
        
        # Major keywords
        major = ['significant', 'substantially', 'major', 'serious', 'marked',
                'requires monitoring', 'dose adjustment', 'toxicity']
        if any(word in desc_lower for word in major):
            return 0.7
        
        # Minor keywords
        minor = ['may increase', 'may decrease', 'slight', 'mild', 'moderate',
                'monitor', 'caution']
        if any(word in desc_lower for word in minor):
            return 0.4
        
        return 0.5  # Default
    
    def _build_node_features(self, use_text_features=True, max_text_features=100):
        """Build feature matrix for all drugs"""
        print("Building node features...")
        
        all_features = []
        
        for drug_id in self.drug_to_idx.keys():
            drug_matches = self.drugs_df[self.drugs_df['drug_id'] == drug_id]
            if len(drug_matches) == 0:
                # Drug not in our dataset, use default features
                features = self._get_default_features()
            else:
                drug_data = drug_matches.iloc[0]
                features = self._extract_drug_features(drug_data)
            all_features.append(features)
        
        # Convert to numpy array
        features_array = np.array(all_features)
        
        # Add text features if requested
        if use_text_features:
            text_features = self._extract_text_features(max_text_features)
            features_array = np.hstack([features_array, text_features])
        
        # Normalize features
        scaler = StandardScaler()
        features_array = scaler.fit_transform(features_array)
        
        # Convert to tensor
        node_features = torch.tensor(features_array, dtype=torch.float)
        
        return node_features
    
    def _extract_drug_features(self, drug_data):
        """Extract numerical features for a single drug"""
        features = []
        
        # Drug type (one-hot)
        features.append(1.0 if drug_data['drug_type'] == 'small_molecule' else 0.0)
        features.append(1.0 if drug_data['drug_type'] == 'biotech' else 0.0)
        
        # State
        features.append(1.0 if drug_data['state'] == 'solid' else 0.0)
        features.append(1.0 if drug_data['state'] == 'liquid' else 0.0)
        
        # Groups (approved, experimental, etc.)
        groups = str(drug_data.get('groups', ''))
        features.append(1.0 if 'approved' in groups else 0.0)
        features.append(1.0 if 'experimental' in groups else 0.0)
        features.append(1.0 if 'withdrawn' in groups else 0.0)
        
        # Text length features (proxy for information richness)
        features.append(len(str(drug_data.get('description', ''))) / 1000.0)
        features.append(len(str(drug_data.get('indication', ''))) / 1000.0)
        
        # Category features (most common categories)
        categories = str(drug_data.get('categories', ''))
        common_categories = [
            'Anticoagulants', 'Antibiotics', 'Analgesics', 'Antidepressants',
            'Antihypertensive', 'Anti-inflammatory', 'Cardiovascular', 'CNS'
        ]
        for cat in common_categories:
            features.append(1.0 if cat.lower() in categories.lower() else 0.0)
        
        return features
    
    def _get_default_features(self):
        """Return default features for drugs not in dataset"""
        # Match the feature count from _extract_drug_features
        num_basic_features = 2 + 2 + 3 + 2  # type + state + groups + text_length
        num_categories = 8  # common categories
        total_features = num_basic_features + num_categories
        return [0.0] * total_features
    
    def _extract_text_features(self, max_features=100):
        """Extract TF-IDF features from drug descriptions"""
        print("  Extracting text features using TF-IDF...")
        
        # Collect all descriptions
        descriptions = []
        for drug_id in self.drug_to_idx.keys():
            drug_matches = self.drugs_df[self.drugs_df['drug_id'] == drug_id]
            if len(drug_matches) == 0:
                descriptions.append('unknown')
            else:
                drug_data = drug_matches.iloc[0]
                desc = str(drug_data.get('description', ''))
                mech = str(drug_data.get('mechanism_of_action', ''))
                text = desc + ' ' + mech
                descriptions.append(text if text.strip() else 'unknown')
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        text_features = vectorizer.fit_transform(descriptions).toarray()
        
        return text_features
    
    def save_graph(self, filepath='data/drug_graph.pt'):
        """Save graph and mappings"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save graph
        torch.save({
            'graph_data': self.graph_data,
            'drug_to_idx': self.drug_to_idx,
            'idx_to_drug': self.idx_to_drug
        }, filepath)
        
        print(f"\n✅ Graph saved to {filepath}")
    
    @staticmethod
    def load_graph(filepath='data/drug_graph.pt'):
        """Load saved graph"""
        data = torch.load(filepath)
        return data['graph_data'], data['drug_to_idx'], data['idx_to_drug']
    
    def get_statistics(self):
        """Get graph statistics"""
        if self.graph_data is None:
            return None
        
        num_edges = self.graph_data.edge_index.shape[1] // 2  # Divide by 2 for undirected
        avg_degree = num_edges * 2 / self.graph_data.num_nodes
        
        stats = {
            'num_nodes': self.graph_data.num_nodes,
            'num_edges': num_edges,
            'avg_degree': avg_degree,
            'num_features': self.graph_data.x.shape[1],
            'density': num_edges / (self.graph_data.num_nodes * (self.graph_data.num_nodes - 1) / 2)
        }
        
        return stats


def main():
    """Example usage"""
    
    # Load parsed data
    print("Loading data...")
    drugs_df = pd.read_csv('data/drugs.csv')
    interactions_df = pd.read_csv('data/interactions.csv')
    
    print(f"Loaded {len(drugs_df)} drugs and {len(interactions_df)} interactions")
    
    # Build graph
    builder = DrugGraphBuilder(drugs_df, interactions_df)
    graph_data = builder.build_graph(use_text_features=True, max_text_features=50)
    
    # Show statistics
    stats = builder.get_statistics()
    print("\n" + "="*60)
    print("GRAPH STATISTICS")
    print("="*60)
    for key, value in stats.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    # Save graph
    builder.save_graph('data/drug_graph.pt')
    
    print("\n✅ Graph construction complete!")
    print("   Ready for GCN training.")


if __name__ == "__main__":
    main()
