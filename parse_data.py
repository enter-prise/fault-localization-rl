import pandas as pd

train_data = pd.read_parquet(
    "data_storage/SWE-bench_Lite/data/dev-00000-of-00001.parquet"
)
test_data = pd.read_parquet(
    "data_storage/SWE-bench_Lite/data/test-00000-of-00001.parquet"
)

print("==== TRAIN COLUMNS ====")
print(train_data.columns.tolist())

print("\n==== TEST COLUMNS ====")
print(test_data.columns.tolist())

print("\n==== ONE TEST SAMPLE (TRANSPOSED) ====")
sample = test_data.iloc[0]
print(sample.to_string())