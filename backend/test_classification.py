#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

from ml_model import ml_risk_identifier
from document_processor import process_document
import tempfile

def test_classification():
    print("Testing document type classification...")

    # Test non-contract text
    non_contract_text = "This is a simple business letter about payment terms and invoice schedules. Please find attached the invoice for services rendered."
    result1 = ml_risk_identifier.classify_document_type(non_contract_text)
    print(f"Non-contract classification: {result1}")

    # Test contract text
    contract_text = "This agreement is made between Company A and Company B for data processing services. The parties agree to comply with GDPR regulations and maintain appropriate security measures."
    result2 = ml_risk_identifier.classify_document_type(contract_text)
    print(f"Contract classification: {result2}")

    # Test with temporary file (unsupported type)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(non_contract_text)
        temp_file = f.name

    try:
        result3 = process_document(temp_file, 'txt')
        print(f"Unsupported file type result: {result3}")
    except Exception as e:
        print(f"Unsupported file type error: {e}")
    finally:
        os.unlink(temp_file)

if __name__ == "__main__":
    test_classification()
