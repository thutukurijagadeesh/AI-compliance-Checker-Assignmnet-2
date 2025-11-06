from document_processor import generate_recommended_clauses

# Test with sample data
document_type = "Data Processing Agreement"
risks = [
    {'risk': 'breach notification', 'severity': 'High'},
    {'risk': 'data processing without consent', 'severity': 'High'},
    {'risk': 'security breach', 'severity': 'Medium'}
]
missing_clauses = [
    {'clause': 'privacy notice'},
    {'clause': 'business associate agreement'},
    {'clause': 'data minimization'}
]

result = generate_recommended_clauses(document_type, risks, missing_clauses)
print('Function executed successfully')
print(f'Generated {len(result)} recommendations')
for i, rec in enumerate(result, 1):
    print(f'{i}. {rec}')
