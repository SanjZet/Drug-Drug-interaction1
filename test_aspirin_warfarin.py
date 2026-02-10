import requests
import json

print("="*70)
print("TESTING ASPIRIN + WARFARIN INTERACTION")
print("="*70)

# Test 1: Search for Warfarin
print("\n1. Searching for Warfarin...")
r = requests.get('http://localhost:5000/api/drugs/search?q=warfarin')
if r.status_code == 200:
    result = r.json()
    print(f"Found {len(result['drugs'])} matches:")
    # Filter out NaN IDs
    valid_drugs = [d for d in result['drugs'] if d['drug_id'] and str(d['drug_id']).lower() != 'nan']
    for drug in valid_drugs[:5]:
        print(f"   {drug['drug_id']}: {drug['name']}")
    warfarin_id = valid_drugs[0]['drug_id'] if valid_drugs else None
else:
    print("Search failed")
    warfarin_id = None

# Test 2: Search for Aspirin  
print("\n2. Searching for Aspirin...")
r = requests.get('http://localhost:5000/api/drugs/search?q=aspirin')
if r.status_code == 200:
    result = r.json()
    print(f"Found {len(result['drugs'])} matches:")
    # Filter out NaN IDs
    valid_drugs = [d for d in result['drugs'] if d['drug_id'] and str(d['drug_id']).lower() != 'nan']
    for drug in valid_drugs:
        print(f"   {drug['drug_id']}: {drug['name']}")
    aspirin_id = valid_drugs[0]['drug_id'] if valid_drugs else None
else:
    print("Search failed")
    aspirin_id = None

# Test 3: Check interaction
if warfarin_id and aspirin_id:
    print(f"\n3. Checking interaction: {aspirin_id} + {warfarin_id}...")
    payload = {'drug1': aspirin_id, 'drug2': warfarin_id}
    r = requests.post('http://localhost:5000/api/interactions/check', json=payload, timeout=120)
    if r.status_code == 200:
        result = r.json()
        print(f"\n✅ INTERACTION RESULT:")
        print(f"   Drug 1: {result['drug1']['name']} ({result['drug1']['id']})")
        print(f"   Drug 2: {result['drug2']['name']} ({result['drug2']['id']})")
        print(f"   Probability: {result['probability']*100:.2f}%")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Source: {result['source']}")
    else:
        print("Interaction check failed:", r.status_code)

print("\n" + "="*70)
print("\n⚠️ NOTE:")
print("   - Pure 'Aspirin' is not in the database")
print("   - Only Nitroaspirin and Carbaspirin calcium are available")
print("   - Warfarin exists as DB00682 and variants")
print("="*70)
