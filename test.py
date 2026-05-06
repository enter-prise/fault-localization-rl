

"""
测试 BM25 Only vs Hybrid (BM25 + Dense) 检索器
"""

from retrieval.bm25 import SimpleBM25
from retrieval.retriever import Retriever
from graph.builder import GraphBuilder


def main():
    print("=" * 60)
    print("检索器测试")
    print("=" * 60)

    # 构建图
    print("\n1. 构建图...")
    builder = GraphBuilder()
    graph = builder.build_from_repo('data_storage/repos/astropy')
    print(f"   图构建完成，共 {len(graph.nodes)} 个节点")

    # 构建文档
    print("\n2. 构建文档...")
    corpus = {}
    for node_id, node in graph.nodes.items():
        name = getattr(node, 'name', '') or ''
        doc = getattr(node, 'doc', '') or ''
        file_path = getattr(node, 'file_path', '') or ''
        code = getattr(node, 'code', '') or ''
        corpus[node_id] = f'{name} {doc} {file_path} {str(code)[:300]}'.strip()

    bm25 = SimpleBM25(corpus)
    print(f"   文档构建完成，共 {len(corpus)} 篇")

    query = 'null pointer exception when accessing database'
    print(f"\n3. 查询: {query}")

    # 测试 BM25 only
    print("\n" + "=" * 60)
    print("测试 BM25 Only")
    print("=" * 60)

    retriever_bm25 = Retriever(bm25, use_dense=False)
    retriever_bm25.build_index(graph)
    results = retriever_bm25.retrieve_with_score(query, top_k=5)

    print("\nTop-5 结果:")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['entity_name']}: {r['relevance_score']:.4f}")

    # 测试混合检索
    print("\n" + "=" * 60)
    print("测试 Hybrid (BM25 + Dense)")
    print("=" * 60)

    retriever_hybrid = Retriever(
        bm25,
        use_dense=True,
        dense_model_path='./dense_retriever_model',
        alpha=0.5,
        beta=0.5,
    )
    retriever_hybrid.build_index(graph)
    results = retriever_hybrid.retrieve_with_score(query, top_k=5)

    print("\nTop-5 结果:")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['entity_name']}: {r['relevance_score']:.4f}")

    # 对比分析
    print("\n" + "=" * 60)
    print("对比分析")
    print("=" * 60)

    # 重新获取结果用于对比
    bm25_results = retriever_bm25.retrieve_with_score(query, top_k=5)
    hybrid_results = retriever_hybrid.retrieve_with_score(query, top_k=5)

    print("\nTop-1 预测:")
    print(f"  BM25 Only:  {bm25_results[0]['entity_name']} (score={bm25_results[0]['relevance_score']:.4f})")
    print(f"  Hybrid:     {hybrid_results[0]['entity_name']} (score={hybrid_results[0]['relevance_score']:.4f})")

    # 检查是否有变化
    if bm25_results[0]['entity_id'] == hybrid_results[0]['entity_id']:
        print("\n  📌 结果一致：Dense 检索没有改变 Top-1 结果")
        print("     建议：适当增加 beta 权重或调整公式")
    else:
        print("\n  🎉 结果不同：Dense 检索改变了排序！")
        print("     BM25 + Dense 提供了不同的检索结果")

    # 显示 ground truth（预期的正确答案是 _cstack 或 separable.py）
    print("\n预期正确答案: _cstack 或 separable.py 或 separability_matrix")
    print("\n✅ 检索器测试完成")


if __name__ == "__main__":
    main()
