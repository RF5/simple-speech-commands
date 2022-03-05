import pandas as pd
from pathlib import Path
import numpy as np
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate train & valid csvs from dataset directories")

    parser.add_argument('--root_path', default=None, type=str)

    args = parser.parse_args()
    rp = Path(args.root_path)
    all_files = list(rp.rglob('**/*.wav'))
    rel_files = [f.relative_to(rp) for f in all_files]

    with open(rp/'validation_list.txt', 'r') as f:
        valid_files = [Path(t.strip()) for t in f.readlines()]
    with open(rp/'testing_list.txt', 'r') as f:
        test_files = [Path(t.strip()) for t in f.readlines()]

    train_files = list( set(rel_files) - set(valid_files) - set(test_files) )
    train_files = [t for t in train_files if '_' not in str(t.parent.stem)]
    train_files = sorted(train_files)
    valid_files = sorted(valid_files)
    test_files = sorted(test_files)

    print(f"All files: {len(rel_files)}. Train: {len(train_files)}, valid: {len(valid_files)}, test: {len(test_files)}.")

    train_df = pd.DataFrame({'path': [str(rp/t) for t in train_files], 'label': [str(t.parent.stem).lower().strip() for t in train_files]})
    valid_df = pd.DataFrame({'path': [str(rp/t) for t in valid_files], 'label': [str(t.parent.stem).lower().strip() for t in valid_files]})
    test_df = pd.DataFrame({'path': [str(rp/t) for t in test_files], 'label': [str(t.parent.stem).lower().strip() for t in test_files]})
    os.makedirs('splits/', exist_ok=True)
    train_df.to_csv("splits/train.csv", index=False)
    valid_df.to_csv("splits/valid.csv", index=False)    
    test_df.to_csv("splits/test.csv", index=False)    

if __name__ == '__main__':
    main()
