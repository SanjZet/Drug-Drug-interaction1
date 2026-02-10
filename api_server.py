"""
Flask API Server for Drug Interaction Checker
Provides REST API endpoints for drug search and interaction checking
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import pandas as pd
from pathlib import Path
from gcn_model import DrugInteractionGCN
from graph_builder import DrugGraphBuilder
import numpy as np

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for frontend access

# Global variables for model and data
model = None
graph_data = None
drug_to_idx = None
idx_to_drug = None
interactions_db = None
drugs_df = None

def load_model_and_data():
    """Load trained model and graph data"""
    global model, graph_data, drug_to_idx, idx_to_drug, interactions_db, drugs_df
    
    print("🔄 Loading model and data...")
    
    # Load trained model
    checkpoint = torch.load('data/trained_model.pt', weights_only=False)
    input_dim = checkpoint['input_dim']
    model = DrugInteractionGCN(input_dim=input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model loaded")
    
    # Load graph data
    graph_dict = torch.load('data/drug_graph.pt', weights_only=False)
    graph_data = graph_dict['graph_data']
    drug_to_idx = graph_dict['drug_to_idx']
    idx_to_drug = graph_dict['idx_to_drug']
    print("✅ Graph data loaded")
    
    # Load interactions database
    interactions_db = pd.read_csv('data/interactions.csv')
    drugs_df = pd.read_csv('data/drugs.csv')
    print("✅ Database loaded")
    
    print(f"📊 {len(drugs_df)} drugs, {len(interactions_db)} interactions")

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'num_drugs': len(drugs_df) if drugs_df is not None else 0,
        'num_interactions': len(interactions_db) if interactions_db is not None else 0
    })

@app.route('/api/drugs', methods=['GET'])
def get_all_drugs():
    """Get list of all drugs"""
    if drugs_df is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    drugs_list = drugs_df[['drug_id', 'name']].to_dict('records')
    return jsonify({
        'drugs': drugs_list,
        'total': len(drugs_list)
    })

@app.route('/api/drugs/search', methods=['GET'])
def search_drugs():
    """Search drugs by name"""
    query = request.args.get('q', '').lower()
    limit = int(request.args.get('limit', 20))
    
    if not query:
        return jsonify({'drugs': [], 'total': 0})
    
    # Search in drug names
    matches = drugs_df[drugs_df['name'].str.lower().str.contains(query, na=False)]
    results = matches[['drug_id', 'name']].head(limit).to_dict('records')
    
    return jsonify({
        'drugs': results,
        'total': len(results),
        'query': query
    })

@app.route('/api/drugs/<drug_id>', methods=['GET'])
def get_drug_details(drug_id):
    """Get detailed information about a specific drug"""
    drug = drugs_df[drugs_df['drug_id'] == drug_id]
    
    if drug.empty:
        return jsonify({'error': 'Drug not found'}), 404
    
    drug_info = drug.iloc[0].to_dict()
    
    # Get known interactions from database
    interactions = interactions_db[
        (interactions_db['drug_id_1'] == drug_id) | 
        (interactions_db['drug_id_2'] == drug_id)
    ]
    
    interaction_list = []
    for _, inter in interactions.iterrows():
        other_drug_id = inter['drug_id_2'] if inter['drug_id_1'] == drug_id else inter['drug_id_1']
        other_drug = drugs_df[drugs_df['drug_id'] == other_drug_id]
        
        if not other_drug.empty:
            interaction_list.append({
                'drug_id': other_drug_id,
                'drug_name': other_drug.iloc[0]['name'],
                'description': inter.get('description', ''),
                'source': 'database',
                'confidence': 1.0
            })
    
    return jsonify({
        'drug': drug_info,
        'interactions': interaction_list,
        'interaction_count': len(interaction_list)
    })

@app.route('/api/interactions/check', methods=['POST'])
def check_interaction():
    """Check interaction between two drugs"""
    data = request.json
    drug1_id = data.get('drug1')
    drug2_id = data.get('drug2')
    
    if not drug1_id or not drug2_id:
        return jsonify({'error': 'Both drug IDs required'}), 400
    
    # Get drug names
    drug1_name = drugs_df[drugs_df['drug_id'] == drug1_id]
    drug2_name = drugs_df[drugs_df['drug_id'] == drug2_id]
    
    if drug1_name.empty or drug2_name.empty:
        return jsonify({'error': 'One or both drugs not found'}), 404
    
    drug1_name = drug1_name.iloc[0]['name']
    drug2_name = drug2_name.iloc[0]['name']
    
    # Check database first
    db_interaction = interactions_db[
        ((interactions_db['drug_id_1'] == drug1_id) & (interactions_db['drug_id_2'] == drug2_id)) |
        ((interactions_db['drug_id_1'] == drug2_id) & (interactions_db['drug_id_2'] == drug1_id))
    ]
    
    if not db_interaction.empty:
        # Known interaction from database
        description = db_interaction.iloc[0].get('description', 'Interaction documented in database')
        return jsonify({
            'drug1': {'id': drug1_id, 'name': drug1_name},
            'drug2': {'id': drug2_id, 'name': drug2_name},
            'interaction_exists': True,
            'probability': 1.0,
            'confidence': 'high',
            'risk_level': 'HIGH',
            'source': 'database',
            'description': description
        })
    
    # Use GCN model for prediction
    if drug1_id not in drug_to_idx or drug2_id not in drug_to_idx:
        return jsonify({
            'drug1': {'id': drug1_id, 'name': drug1_name},
            'drug2': {'id': drug2_id, 'name': drug2_name},
            'interaction_exists': False,
            'probability': 0.0,
            'confidence': 'unknown',
            'risk_level': 'UNKNOWN',
            'source': 'model',
            'description': 'One or both drugs not in training data'
        })
    
    idx1 = drug_to_idx[drug1_id]
    idx2 = drug_to_idx[drug2_id]
    
    # Get node embeddings
    with torch.no_grad():
        x = graph_data.x
        edge_index = graph_data.edge_index
        embeddings = model.encode(x, edge_index)
        
        # Create edge tensor for the pair
        test_edge = torch.tensor([[idx1], [idx2]], dtype=torch.long)
        logit = model.decode(embeddings, test_edge)
        probability = torch.sigmoid(logit).item()
    
    # Classify risk level
    if probability > 0.7:
        risk_level = 'HIGH'
        confidence = 'high'
    elif probability > 0.5:
        risk_level = 'MEDIUM'
        confidence = 'medium'
    elif probability > 0.3:
        risk_level = 'LOW'
        confidence = 'medium'
    else:
        risk_level = 'VERY LOW'
        confidence = 'low'
    
    return jsonify({
        'drug1': {'id': drug1_id, 'name': drug1_name},
        'drug2': {'id': drug2_id, 'name': drug2_name},
        'interaction_exists': probability > 0.5,
        'probability': round(probability, 4),
        'confidence': confidence,
        'risk_level': risk_level,
        'source': 'model',
        'description': f'GCN model prediction: {probability*100:.2f}% probability of interaction'
    })

@app.route('/api/interactions/batch', methods=['POST'])
def check_batch_interactions():
    """Check interactions for multiple drug pairs"""
    data = request.json
    drug_ids = data.get('drugs', [])
    
    if len(drug_ids) < 2:
        return jsonify({'error': 'At least 2 drugs required'}), 400
    
    results = []
    
    # Check all pairs
    for i in range(len(drug_ids)):
        for j in range(i + 1, len(drug_ids)):
            drug1_id = drug_ids[i]
            drug2_id = drug_ids[j]
            
            # Get drug names
            drug1 = drugs_df[drugs_df['drug_id'] == drug1_id]
            drug2 = drugs_df[drugs_df['drug_id'] == drug2_id]
            
            if drug1.empty or drug2.empty:
                continue
            
            drug1_name = drug1.iloc[0]['name']
            drug2_name = drug2.iloc[0]['name']
            
            # Check database
            db_interaction = interactions_db[
                ((interactions_db['drug_id_1'] == drug1_id) & (interactions_db['drug_id_2'] == drug2_id)) |
                ((interactions_db['drug_id_1'] == drug2_id) & (interactions_db['drug_id_2'] == drug1_id))
            ]
            
            if not db_interaction.empty:
                description = db_interaction.iloc[0].get('description', 'Known interaction')
                results.append({
                    'drug1': {'id': drug1_id, 'name': drug1_name},
                    'drug2': {'id': drug2_id, 'name': drug2_name},
                    'probability': 1.0,
                    'risk_level': 'HIGH',
                    'source': 'database',
                    'description': description
                })
            elif drug1_id in drug_to_idx and drug2_id in drug_to_idx:
                # Use model
                idx1 = drug_to_idx[drug1_id]
                idx2 = drug_to_idx[drug2_id]
                
                with torch.no_grad():
                    x = graph_data.x
                    edge_index = graph_data.edge_index
                    embeddings = model.encode(x, edge_index)
                    test_edge = torch.tensor([[idx1], [idx2]], dtype=torch.long)
                    logit = model.decode(embeddings, test_edge)
                    probability = torch.sigmoid(logit).item()
                
                if probability > 0.5:
                    risk_level = 'HIGH' if probability > 0.7 else 'MEDIUM'
                    results.append({
                        'drug1': {'id': drug1_id, 'name': drug1_name},
                        'drug2': {'id': drug2_id, 'name': drug2_name},
                        'probability': round(probability, 4),
                        'risk_level': risk_level,
                        'source': 'model',
                        'description': f'Predicted interaction ({probability*100:.1f}%)'
                    })
    
    return jsonify({
        'interactions': results,
        'total_checked': len(drug_ids),
        'interactions_found': len(results)
    })

@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Get system statistics"""
    return jsonify({
        'total_drugs': len(drugs_df),
        'total_interactions': len(interactions_db),
        'model_nodes': len(idx_to_drug),
        'model_ready': model is not None
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("DRUG INTERACTION CHECKER - API SERVER".center(60))
    print("="*60 + "\n")
    
    # Load model and data
    load_model_and_data()
    
    print("\n" + "="*60)
    print("🚀 Starting Flask server...")
    print("="*60)
    print("\n📡 API Endpoints:")
    print("   GET  /api/health              - Health check")
    print("   GET  /api/drugs               - List all drugs")
    print("   GET  /api/drugs/search?q=name - Search drugs")
    print("   GET  /api/drugs/<id>          - Drug details")
    print("   POST /api/interactions/check  - Check interaction")
    print("   POST /api/interactions/batch  - Batch check")
    print("   GET  /api/stats               - Statistics")
    print("\n🌐 Server running at: http://localhost:5000")
    print("   Frontend: http://localhost:5000")
    print("   API Docs: http://localhost:5000/api/health\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
