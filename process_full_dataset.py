"""
Process Full DrugBank Dataset
This script processes the complete DrugBank XML database without drug limit
"""

import sys
from pathlib import Path
from data_parser import DrugBankParser
from graph_builder import DrugGraphBuilder
from gcn_model import GCNTrainer

def process_full_dataset():
    """Process the complete DrugBank dataset"""
    
    print("\n" + "="*70)
    print("PROCESSING FULL DRUGBANK DATASET".center(70))
    print("="*70 + "\n")
    
    print("⚠️  WARNING: This will process the entire DrugBank database")
    print("   Expected: ~15,000 drugs and millions of interactions")
    print("   This may take 30-60 minutes and use significant RAM\n")
    
    response = input("Continue? (yes/no): ").strip().lower()
    if response != 'yes':
        print("❌ Cancelled by user")
        return
    
    # Step 1: Parse XML (no drug limit)
    print("\n" + "="*70)
    print("[STEP 1/3] PARSING DRUGBANK XML - FULL DATASET".center(70))
    print("="*70 + "\n")
    
    xml_path = Path("../full database.xml")
    if not xml_path.exists():
        print(f"❌ Error: Database file not found: {xml_path}")
        return
    
    parser = DrugBankParser(xml_path)
    
    print("📊 Starting full dataset parsing...")
    print("   This will take several minutes...\n")
    
    # Parse without drug limit
    drugs_df, interactions_df, categories_df = parser.parse(
        max_drugs=None,  # Process ALL drugs
        output_dir='data_full'
    )
    
    print(f"\n✅ Parsing complete!")
    print(f"   Total drugs extracted: {len(drugs_df):,}")
    print(f"   Total interactions: {len(interactions_df):,}")
    
    # Step 2: Build graph
    print("\n" + "="*70)
    print("[STEP 2/3] BUILDING DRUG INTERACTION GRAPH".center(70))
    print("="*70 + "\n")
    
    builder = DrugGraphBuilder()
    graph_data, drug_to_idx, idx_to_drug = builder.build_graph(
        drugs_csv='data_full/drugs.csv',
        interactions_csv='data_full/interactions.csv',
        output_path='data_full/drug_graph.pt'
    )
    
    print(f"\n✅ Graph construction complete!")
    print(f"   Nodes: {graph_data.x.shape[0]:,}")
    print(f"   Edges: {graph_data.edge_index.shape[1]:,}")
    print(f"   Features per node: {graph_data.x.shape[1]}")
    
    # Step 3: Train GCN model
    print("\n" + "="*70)
    print("[STEP 3/3] TRAINING GCN MODEL ON FULL DATASET".center(70))
    print("="*70 + "\n")
    
    print("⚠️  Training on full dataset may take 1-2 hours")
    response = input("Train model now? (yes/no): ").strip().lower()
    
    if response == 'yes':
        trainer = GCNTrainer(
            input_dim=graph_data.x.shape[1],
            hidden_dim=256,
            embedding_dim=128
        )
        
        # Train model
        best_model, train_metrics, val_metrics, test_metrics = trainer.train(
            graph_data,
            epochs=200,
            patience=20,
            output_dir='data_full'
        )
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE".center(70))
        print("="*70)
        print(f"\n📊 Final Test Metrics:")
        print(f"   AUC: {test_metrics['auc']:.4f}")
        print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   Precision: {test_metrics['precision']:.4f}")
        print(f"   Recall: {test_metrics['recall']:.4f}")
        print(f"   F1 Score: {test_metrics['f1']:.4f}")
    else:
        print("⏭️  Skipping model training")
    
    print("\n" + "="*70)
    print("FULL DATASET PROCESSING COMPLETE!".center(70))
    print("="*70)
    print("\n📁 Output files saved in 'data_full/' folder:")
    print("   ✅ drugs.csv - All drug information")
    print("   ✅ interactions.csv - All interactions")
    print("   ✅ drug_graph.pt - Complete graph structure")
    if response == 'yes':
        print("   ✅ trained_model.pt - Trained GCN model")
        print("   ✅ best_model.pt - Best checkpoint")
        print("   ✅ training_curves.png - Training visualization")
    
    print("\n💡 To use the full dataset with the web interface:")
    print("   1. Replace 'data/' with 'data_full/' in api_server.py")
    print("   2. Restart the Flask server")
    print("   3. Access http://localhost:5000\n")

def quick_stats():
    """Show quick statistics about current vs full dataset"""
    print("\n" + "="*70)
    print("DATASET COMPARISON".center(70))
    print("="*70 + "\n")
    
    # Current dataset
    try:
        import pandas as pd
        current_drugs = pd.read_csv('data/drugs.csv')
        current_interactions = pd.read_csv('data/interactions.csv')
        
        print("📊 Current Dataset (Quick Test):")
        print(f"   Drugs: {len(current_drugs):,}")
        print(f"   Interactions: {len(current_interactions):,}")
    except:
        print("❌ Current dataset not found")
    
    # Full dataset (if exists)
    try:
        full_drugs = pd.read_csv('data_full/drugs.csv')
        full_interactions = pd.read_csv('data_full/interactions.csv')
        
        print("\n📊 Full Dataset:")
        print(f"   Drugs: {len(full_drugs):,}")
        print(f"   Interactions: {len(full_interactions):,}")
        
        if len(current_drugs) > 0:
            drug_ratio = len(full_drugs) / len(current_drugs)
            inter_ratio = len(full_interactions) / len(current_interactions)
            print(f"\n📈 Full dataset is:")
            print(f"   {drug_ratio:.1f}x more drugs")
            print(f"   {inter_ratio:.1f}x more interactions")
    except:
        print("\n⚠️  Full dataset not yet processed")
        print("   Run this script to process it!")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Process full DrugBank dataset')
    parser.add_argument('--stats', action='store_true', 
                       help='Show dataset statistics only')
    parser.add_argument('--force', action='store_true',
                       help='Skip confirmation prompts')
    
    args = parser.parse_args()
    
    if args.stats:
        quick_stats()
    else:
        process_full_dataset()
