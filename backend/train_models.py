#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

from ml_model import ml_extractor

def main():
    print("Starting model training...")
    try:
        ml_extractor.train_models()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
