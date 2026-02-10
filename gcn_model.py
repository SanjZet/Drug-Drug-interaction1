"""
Drug Interaction Project - GCN Model
Graph Convolutional Network for drug interaction prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import negative_sampling
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from tqdm import tqdm

class DrugInteractionGCN(nn.Module):
    """
    Graph Convolutional Network for Drug Interaction Prediction
    Uses link prediction approach
    """
    
    def __init__(self, input_dim, hidden_dim=256, embedding_dim=128, dropout=0.5):
        super(DrugInteractionGCN, self).__init__()
        
        # GCN layers for encoding drugs
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, embedding_dim)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Link prediction decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def encode(self, x, edge_index):
        """
        Encode drugs into embedding space using GCN
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge list [2, num_edges]
        
        Returns:
            Drug embeddings [num_nodes, embedding_dim]
        """
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 3
        x = self.conv3(x, edge_index)
        
        return x
    
    def decode(self, z, edge_index):
        """
        Predict interaction probability for drug pairs
        
        Args:
            z: Drug embeddings [num_nodes, embedding_dim]
            edge_index: Pairs to predict [2, num_pairs]
        
        Returns:
            Interaction probabilities [num_pairs, 1]
        """
        # Get embeddings for each drug in pair
        drug1_embed = z[edge_index[0]]
        drug2_embed = z[edge_index[1]]
        
        # Concatenate pair embeddings
        pair_embed = torch.cat([drug1_embed, drug2_embed], dim=1)
        
        # Predict interaction
        logits = self.decoder(pair_embed)
        
        return logits
    
    def forward(self, x, edge_index, edge_label_index):
        """
        Full forward pass
        
        Args:
            x: Node features
            edge_index: Training edges (message passing)
            edge_label_index: Edges to predict
        
        Returns:
            Predictions for edge_label_index
        """
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)
    
    def predict_interaction(self, x, edge_index, drug1_idx, drug2_idx):
        """
        Predict interaction between two specific drugs
        
        Args:
            x: Node features
            edge_index: Graph edges
            drug1_idx: Index of first drug
            drug2_idx: Index of second drug
        
        Returns:
            Interaction probability (0-1)
        """
        self.eval()
        with torch.no_grad():
            z = self.encode(x, edge_index)
            test_edge = torch.tensor([[drug1_idx], [drug2_idx]], dtype=torch.long)
            logit = self.decode(z, test_edge)
            prob = torch.sigmoid(logit).item()
        return prob


class GCNTrainer:
    """Trainer for Drug Interaction GCN"""
    
    def __init__(self, model, graph_data, device='cpu'):
        self.model = model.to(device)
        self.graph_data = graph_data.to(device)
        self.device = device
        self.train_losses = []
        self.val_aucs = []
    
    def split_edges(self, train_ratio=0.8, val_ratio=0.1):
        """
        Split edges into train/val/test sets
        
        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
        """
        edge_index = self.graph_data.edge_index
        num_edges = edge_index.shape[1] // 2  # Undirected, so divide by 2
        
        # Get unique edges (remove reverse edges)
        edge_list = edge_index.t().tolist()
        unique_edges = []
        seen = set()
        for e in edge_list:
            edge_tuple = tuple(sorted(e))
            if edge_tuple not in seen:
                unique_edges.append(e)
                seen.add(edge_tuple)
        
        unique_edges = torch.tensor(unique_edges).t()
        
        # Shuffle
        perm = torch.randperm(unique_edges.shape[1])
        unique_edges = unique_edges[:, perm]
        
        # Split
        num_train = int(train_ratio * unique_edges.shape[1])
        num_val = int(val_ratio * unique_edges.shape[1])
        
        self.train_edges = unique_edges[:, :num_train]
        self.val_edges = unique_edges[:, num_train:num_train+num_val]
        self.test_edges = unique_edges[:, num_train+num_val:]
        
        # Add reverse edges for message passing
        self.train_edges_undirected = torch.cat([
            self.train_edges,
            torch.stack([self.train_edges[1], self.train_edges[0]])
        ], dim=1)
        
        print(f"Edge split:")
        print(f"  Train: {self.train_edges.shape[1]} edges")
        print(f"  Val: {self.val_edges.shape[1]} edges")
        print(f"  Test: {self.test_edges.shape[1]} edges")
    
    def train_epoch(self, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        optimizer.zero_grad()
        
        # Positive samples
        pos_edge_index = self.train_edges
        
        # Negative sampling
        neg_edge_index = negative_sampling(
            edge_index=self.train_edges_undirected,
            num_nodes=self.graph_data.num_nodes,
            num_neg_samples=pos_edge_index.shape[1]
        )
        
        # Combine positive and negative
        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        edge_labels = torch.cat([
            torch.ones(pos_edge_index.shape[1], 1),
            torch.zeros(neg_edge_index.shape[1], 1)
        ], dim=0).to(self.device)
        
        # Forward pass
        logits = self.model(
            self.graph_data.x,
            self.train_edges_undirected,
            edge_label_index
        )
        
        # Loss
        loss = criterion(logits, edge_labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def evaluate(self, edge_index):
        """Evaluate on given edge set"""
        self.model.eval()
        
        with torch.no_grad():
            # Positive samples
            pos_edge_index = edge_index
            
            # Negative samples
            neg_edge_index = negative_sampling(
                edge_index=self.train_edges_undirected,
                num_nodes=self.graph_data.num_nodes,
                num_neg_samples=pos_edge_index.shape[1]
            )
            
            # Combine
            edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            edge_labels = torch.cat([
                torch.ones(pos_edge_index.shape[1]),
                torch.zeros(neg_edge_index.shape[1])
            ]).cpu().numpy()
            
            # Predict
            logits = self.model(
                self.graph_data.x,
                self.train_edges_undirected,
                edge_label_index
            )
            
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            # Metrics
            auc = roc_auc_score(edge_labels, probs)
            acc = accuracy_score(edge_labels, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                edge_labels, preds, average='binary'
            )
        
        return {
            'auc': auc,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, epochs=100, lr=0.001, weight_decay=5e-4):
        """
        Full training loop
        
        Args:
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: L2 regularization
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_auc = 0.0
        patience = 20
        patience_counter = 0
        
        print("\nStarting training...")
        for epoch in tqdm(range(epochs)):
            # Train
            loss = self.train_epoch(optimizer, criterion)
            self.train_losses.append(loss)
            
            # Validate every 5 epochs
            if (epoch + 1) % 5 == 0:
                val_metrics = self.evaluate(self.val_edges)
                self.val_aucs.append(val_metrics['auc'])
                
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"  Loss: {loss:.4f}")
                print(f"  Val AUC: {val_metrics['auc']:.4f}")
                print(f"  Val Acc: {val_metrics['accuracy']:.4f}")
                print(f"  Val F1: {val_metrics['f1']:.4f}")
                
                # Early stopping
                if val_metrics['auc'] > best_val_auc:
                    best_val_auc = val_metrics['auc']
                    patience_counter = 0
                    torch.save(self.model.state_dict(), 'data/best_model.pt')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('data/best_model.pt'))
        
        # Final test evaluation
        print("\n" + "="*60)
        print("FINAL TEST RESULTS")
        print("="*60)
        test_metrics = self.evaluate(self.test_edges)
        for metric, value in test_metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        return test_metrics
    
    def plot_training(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curve
        ax1.plot(self.train_losses)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True)
        
        # AUC curve
        ax2.plot(self.val_aucs)
        ax2.set_xlabel('Validation Step')
        ax2.set_ylabel('AUC')
        ax2.set_title('Validation AUC')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('data/training_curves.png', dpi=150)
        print("Training curves saved to data/training_curves.png")


def main():
    """Example usage"""
    from graph_builder import DrugGraphBuilder
    
    # Load graph
    print("Loading graph...")
    graph_data, drug_to_idx, idx_to_drug = DrugGraphBuilder.load_graph('data/drug_graph.pt')
    
    # Initialize model
    input_dim = graph_data.x.shape[1]
    model = DrugInteractionGCN(input_dim, hidden_dim=256, embedding_dim=128)
    
    print(f"\nModel architecture:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: 256")
    print(f"  Embedding dim: 128")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    trainer = GCNTrainer(model, graph_data, device=device)
    trainer.split_edges(train_ratio=0.8, val_ratio=0.1)
    
    test_metrics = trainer.train(epochs=200, lr=0.001)
    
    # Plot
    trainer.plot_training()
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'drug_to_idx': drug_to_idx,
        'idx_to_drug': idx_to_drug,
        'input_dim': input_dim
    }, 'data/trained_model.pt')
    
    print("\n✅ Training complete! Model saved to data/trained_model.pt")


if __name__ == "__main__":
    main()
