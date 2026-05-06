#!/usr/bin/env python3
"""
SWE-bench Lite 数据划分脚本
按仓库划分，避免数据泄露
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():
    # 加载数据
    df = pd.read_parquet('data_storage/SWE-bench_Lite/data/test-00000-of-00001.parquet')
    print(f"原始数据: {len(df)} 个样本")
    
    # 获取唯一仓库
    repos = df['repo'].unique()
    print(f"唯一仓库数: {len(repos)}")
    print("\n仓库列表:")
    for r in repos:
        count = len(df[df['repo'] == r])
        print(f"  {r}: {count} 个样本")
    
    # 划分: 60% 训练, 20% 验证, 20% 测试
    train_repos, temp_repos = train_test_split(
        repos, 
        train_size=0.6,  # 60% 训练
        random_state=42
    )
    
    # 剩余 40% 平分给验证和测试 (各20%)
    val_repos, test_repos = train_test_split(
        temp_repos,
        train_size=0.5,  # 20% 验证, 20% 测试
        random_state=42
    )
    
    # 根据仓库划分数据
    train_df = df[df['repo'].isin(train_repos)]
    val_df = df[df['repo'].isin(val_repos)]
    test_df = df[df['repo'].isin(test_repos)]
    
    # 保存
    os.makedirs('data_storage/splits', exist_ok=True)
    train_df.to_parquet('data_storage/splits/train.parquet')
    val_df.to_parquet('data_storage/splits/val.parquet')
    test_df.to_parquet('data_storage/splits/test.parquet')
    
    print("\n" + "=" * 60)
    print("划分结果")
    print("=" * 60)
    print(f"训练集: {len(train_df)} 个样本 ({len(train_repos)} 个仓库)")
    print(f"验证集: {len(val_df)} 个样本 ({len(val_repos)} 个仓库)")
    print(f"测试集: {len(test_df)} 个样本 ({len(test_repos)} 个仓库)")
    
    print("\n训练集仓库:")
    for r in train_repos:
        print(f"  {r}: {len(train_df[train_df['repo'] == r])} 个样本")
    
    print("\n验证集仓库:")
    for r in val_repos:
        print(f"  {r}: {len(val_df[val_df['repo'] == r])} 个样本")
    
    print("\n测试集仓库:")
    for r in test_repos:
        print(f"  {r}: {len(test_df[test_df['repo'] == r])} 个样本")
    
    # 保存划分信息到 JSON
    import json
    split_info = {
        "train": {"repos": list(train_repos), "samples": len(train_df)},
        "val": {"repos": list(val_repos), "samples": len(val_df)},
        "test": {"repos": list(test_repos), "samples": len(test_df)},
        "total": len(df)
    }
    with open('data_storage/splits/split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print("\n✅ 保存到 data_storage/splits/")
    print("   - train.parquet")
    print("   - val.parquet")
    print("   - test.parquet")
    print("   - split_info.json")

if __name__ == "__main__":
    main()