#!/usr/bin/env python3
"""测试 ground truth 提取是否正确"""

import pandas as pd
import re

def extract_ground_truth(sample):
    ground_truth = []
    
    if "patch" in sample and sample["patch"]:
        patch = sample["patch"]
        
        # 提取文件路径
        file_matches = re.findall(r'diff --git a/(.+?) b/(.+?)\n', patch)
        for old_file, new_file in file_matches:
            ground_truth.append(new_file)
            file_name = new_file.split('/')[-1].replace('.py', '')
            ground_truth.append(file_name)
        
        # 提取函数名
        func_matches = re.findall(r'@@.*@@\s+def\s+(\w+)\s*\(', patch)
        for func in func_matches:
            ground_truth.append(func)
    
    return list(set(ground_truth))

# 测试
df = pd.read_parquet('data_storage/SWE-bench_Lite/data/test-00000-of-00001.parquet')
sample = df.iloc[0]

ground_truth = extract_ground_truth(sample)
print("Sample 0 ground truth:")
for gt in ground_truth:
    print(f"  - {gt}")

print("\n期望包含: separable.py, _cstack")