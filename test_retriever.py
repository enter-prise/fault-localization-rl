from graph.builder import GraphBuilder
from retrieval.retriever import Retriever

repo_path = "data_storage/repos/astropy"

builder = GraphBuilder()
graph = builder.build_from_repo(repo_path)

retriever = Retriever(graph)
retriever.build_index()

query = "pytest configuration header"
results = retriever.retrieve_with_score(query, top_k=5)

print("=== Retrieval Results ===")
for node_id, score in results:
    info = retriever.inspect_result(node_id)
    print(f"\nID: {node_id}, Score: {score:.4f}")
    print(f"Name: {info['name']}")
    print(f"Type: {info['type']}")
    print(f"Path: {info['file_path']}")
    print(f"Code Preview: {info['code_preview'][:150]}")