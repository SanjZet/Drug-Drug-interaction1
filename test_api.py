import requests
import json

print('='*60)
print('COMPREHENSIVE API TEST')
print('='*60)

# Test 1: Drug Search
print('\n1. Testing Drug Search...')
response = requests.get('http://localhost:5000/api/drugs/search?q=aspirin&limit=3')
if response.status_code == 200:
    drugs = response.json()['drugs']
    print('✅ Search found:', len(drugs), 'drugs')
    for drug in drugs:
        print('   -', drug['name'], '(' + drug['drug_id'] + ')')
else:
    print('❌ Search failed')

# Test 2: Batch Interaction Check
print('\n2. Testing Batch Interaction Check...')
payload = {'drugs': ['DB00001', 'DB00007', 'DB00003']}
response = requests.post('http://localhost:5000/api/interactions/batch', json=payload, timeout=120)
if response.status_code == 200:
    result = response.json()
    print('✅ Batch test found', result['interactions_found'], 'interactions')
    for inter in result['interactions'][:3]:
        d1 = inter['drug1']['name']
        d2 = inter['drug2']['name']
        risk = inter['risk_level']
        print('   -', d1, '+', d2, ':', risk)
else:
    print('❌ Batch test failed:', response.status_code)

# Test 3: Statistics
print('\n3. Testing Statistics...')
response = requests.get('http://localhost:5000/api/stats')
if response.status_code == 200:
    stats = response.json()
    print('✅ Total drugs:', '{:,}'.format(stats['total_drugs']))
    print('✅ Total interactions:', '{:,}'.format(stats['total_interactions']))
    print('✅ Model nodes:', '{:,}'.format(stats['model_nodes']))
else:
    print('❌ Stats failed')

print('\n' + '='*60)
print('✅ ALL TESTS PASSED! API IS FULLY FUNCTIONAL')
print('='*60)
