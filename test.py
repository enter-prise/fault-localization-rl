import pandas as pd

df = pd.read_parquet('/home/kemove/Zhang/fault_localization_rl/data_storage/SWE-bench_Lite/data/test-00000-of-00001.parquet')

# 筛选目标实例
example = df[df['instance_id'] == 'django__django-15202'].iloc[0]

print("=" * 60)
print("实例 ID:", example['instance_id'])
print("=" * 60)
print("仓库:", example['repo'])
print("=" * 60)
print("问题描述 (故障现象):")
print(example['problem_statement'])
print("=" * 60)
print("修复补丁 (正确答案):")
print(example['patch'])