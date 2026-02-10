"""
Demo: Test the Trained GCN Model
Quick demonstration of drug interaction prediction
"""

import torch
import pandas as pd
from gcn_model import DrugInteractionGCN
from graph_builder import DrugGraphBuilder

def main():
    print("="*70)
    print("DRUG INTERACTION PREDICTION - DEMO")
    print("Using Trained GCN Model")
    print("="*70)
    
    # Load trained model
    print("\n📦 Loading trained model...")
    checkpoint = torch.load('data/trained_model.pt')
    
    input_dim = checkpoint['input_dim']
    drug_to_idx = checkpoint['drug_to_idx']
    idx_to_drug = checkpoint['idx_to_drug']
    
    model = DrugInteractionGCN(input_dim=input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded successfully")
    if 'test_metrics' in checkpoint:
        print(f"   Test AUC: {checkpoint['test_metrics']['auc']:.4f}")
        print(f"   Test Accuracy: {checkpoint['test_metrics']['accuracy']:.4f}")
        print(f"   Test F1: {checkpoint['test_metrics']['f1']:.4f}")
    else:
        print(f"   Model trained and ready for predictions")
    
    # Load graph
    print("\n📊 Loading graph data...")
    graph_data, _, _ = DrugGraphBuilder.load_graph('data/drug_graph.pt')
    print(f"✅ Graph loaded: {graph_data.num_nodes} nodes, {graph_data.edge_index.shape[1]//2} edges")
    
    # Load drug names
    drugs_df = pd.read_csv('data/drugs.csv')
    
    # Get list of available drugs
    available_drugs = []
    for drug_id in drug_to_idx.keys():
        drug_match = drugs_df[drugs_df['drug_id'] == drug_id]
        if len(drug_match) > 0:
            available_drugs.append({
                'drug_id': drug_id,
                'name': drug_match.iloc[0]['name'],
                'idx': drug_to_idx[drug_id]
            })
    
    print(f"\n📋 Available drugs in model: {len(available_drugs)}")
    print("\nFirst 10 drugs:")
    for i, drug in enumerate(available_drugs[:10], 1):
        print(f"  {i}. {drug['name']} ({drug['drug_id']})")
    
    # Test some predictions
    print("\n" + "="*70)
    print("TESTING PREDICTIONS")
    print("="*70)
    
    # Test random pairs
    import random
    num_tests = 5
    
    print(f"\nTesting {num_tests} random drug pairs:\n")
    
    tested_pairs = set()
    test_count = 0
    
    while test_count < num_tests and len(tested_pairs) < len(available_drugs):
        # Randomly select two drugs
        drug1, drug2 = random.sample(available_drugs, 2)
        
        pair_key = tuple(sorted([drug1['drug_id'], drug2['drug_id']]))
        if pair_key in tested_pairs:
            continue
        tested_pairs.add(pair_key)
        
        # Predict interaction
        prob = model.predict_interaction(
            graph_data.x,
            graph_data.edge_index,
            drug1['idx'],
            drug2['idx']
        )
        
        # Determine risk level
        if prob > 0.7:
            risk = "🔴 HIGH"
        elif prob > 0.5:
            risk = "🟡 MEDIUM"
        else:
            risk = "🟢 LOW"
        
        print(f"Test {test_count + 1}:")
        print(f"  {drug1['name']} ↔ {drug2['name']}")
        print(f"  Interaction Probability: {prob:.2%}")
        print(f"  Risk Level: {risk}")
        print()
        
        test_count += 1
    
    print("="*70)
    print("✅ DEMO COMPLETE!")
    print("="*70)
    print("\n📝 Next Steps:")
    print("  1. Build web API (api_server.py)")
    print("  2. Create frontend interface")
    print("  3. Add more visualizations")
    print("  4. Deploy to cloud")
    print("\n💡 Your GCN model is ready to use for drug interaction prediction!")

if __name__ == "__main__":
    main()
