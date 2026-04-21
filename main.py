import argparse
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from stable_baselines3 import PPO
import numpy as np 
from graph.builder import GraphBuilder
from retrieval.bm25 import SimpleBM25
from retrieval.retriever import Retriever
from agent.rl_env import FaultLocalizationEnv
from verifier.debate_agent import VerifierAgent
from reasoner.reasoner_agent import ReasonerAgent
from reasoner.snippet_refiner import SnippetRefiner


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def safe_attr(obj, attr: str, default=None):
    return getattr(obj, attr, default)


def load_swebench_sample(sample_idx: int, parquet_path: str) -> Dict[str, Any]:
    df = pd.read_parquet(parquet_path)
    if sample_idx < 0 or sample_idx >= len(df):
        raise IndexError(f"sample_idx={sample_idx} out of range. Dataset size={len(df)}")
    sample = df.iloc[sample_idx]
    return dict(sample)


def build_graph_and_retriever(repo_path: str):
    print(f"Building graph from {repo_path}...")
    builder = GraphBuilder()
    graph = builder.build_from_repo(repo_path)

    corpus = {}
    for node_id, node in graph.nodes.items():
        name = safe_attr(node, "name", "") or ""
        doc = safe_attr(node, "doc", "") or ""
        file_path = safe_attr(node, "file_path", "") or ""
        code = safe_attr(node, "code", "") or ""
        corpus[node_id] = f"{name} {doc} {file_path} {str(code)[:300]}".strip()

    bm25 = SimpleBM25(corpus)
    retriever = Retriever(bm25)

    # 如果你的 Retriever 有 build_index(graph)，就用；否则忽略
    try:
        retriever.build_index(graph)
    except Exception:
        pass

    return graph, retriever


def retrieve_candidates(retriever, graph, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
    """
    兼容不同 retriever.retrieve_with_score 接口：
    1) retrieve_with_score(query, top_k=top_k)
    2) retrieve_with_score(query, graph)
    """
    try:
        results = retriever.retrieve_with_score(query, top_k=top_k)
        return results[:top_k]
    except TypeError:
        results = retriever.retrieve_with_score(query, graph)
        return results[:top_k]


def serialize_candidate(graph, node_id: str, score: float, rank: int, verifier_result: Optional[Dict] = None):
    node = graph.nodes.get(node_id)
    if node is None:
        return {
            "rank": rank,
            "node_id": node_id,
            "score": round(float(score), 4),
            "entity_name": "unknown",
            "entity_type": "unknown",
            "file_path": "",
            "line_number": None,
            "verifier_verdict": None,
            "evidence": [],
            "code_preview": "",
        }

    return {
        "rank": rank,
        "node_id": node_id,
        "score": round(float(score), 4),
        "entity_name": safe_attr(node, "name", "unknown"),
        "entity_type": safe_attr(node, "type", "unknown"),
        "file_path": safe_attr(node, "file_path", ""),
        "line_number": safe_attr(node, "line_number", None),
        "verifier_verdict": verifier_result.get("verdict") if verifier_result else None,
        "evidence": verifier_result.get("evidence", [])[:8] if verifier_result else [],
        "code_preview": str(safe_attr(node, "code", "") or "")[:300],
    }


# ----------------------------------------------------------------------
# Train mode
# ----------------------------------------------------------------------

def train_mode(args):
    """
    训练模式
    """
    print("=" * 60)
    print("Starting TRAINING mode")
    print("=" * 60)

    sample = load_swebench_sample(args.sample_idx, args.data_path)
    issue = sample.get("problem_statement", "")
    repo_path = args.repo_path

    print(f"Training sample_idx={args.sample_idx}")
    print(f"Issue preview: {issue[:120]}")

    graph, retriever = build_graph_and_retriever(repo_path)

    verifier = VerifierAgent(graph, use_llm=args.use_llm)
    reasoner = ReasonerAgent(graph, retriever, use_llm=args.use_llm)

    env = FaultLocalizationEnv(
        graph=graph,
        retriever=retriever,
        bug_query=issue,
        bug_node=args.bug_node if args.bug_node else None,
        max_steps=args.max_steps,
        top_k_retrieval=args.top_k_retrieval,
        query_mode=args.query_mode,
        reasoner=reasoner,
        verifier=verifier,
        seed=args.seed,
    )
    
    # 打印环境信息确认
    print(f"Observation space: {env.observation_space}")  # 应该是 (32,)
    print(f"Action space: {env.action_space}")  # 应该是 MultiDiscrete

    print("Starting PPO training...")
    
    # 对于 MultiDiscrete 动作空间，使用 MultiInputPolicy 或 MlpPolicy
    # PPO 会自动处理 MultiDiscrete
    model = PPO(
        "MlpPolicy",  # MlpPolicy 可以处理 Box 观察空间和 MultiDiscrete 动作空间
        env,
        verbose=1,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        batch_size=args.batch_size,
        n_steps=args.n_steps if hasattr(args, 'n_steps') else 2048,
        n_epochs=args.n_epochs if hasattr(args, 'n_epochs') else 10,
        gamma=args.gamma if hasattr(args, 'gamma') else 0.99,
        device=args.device,
        tensorboard_log=args.tensorboard_log,
        policy_kwargs=dict(
            net_arch=[128, 128],  # 对于 32 维输入，使用更大的网络
        )
    )

    model.learn(total_timesteps=args.timesteps)
    model.save(args.model_output)

    print(f"Model saved to {args.model_output}")
    print("Training completed.")


# ----------------------------------------------------------------------
# Inference mode
# ----------------------------------------------------------------------

def inference_mode(args):
    """
    推理模式
    输入：
        - repo
        - issue text
        - trained PPO model（可选但建议有）

    输出：
        - 导航轨迹
        - top-k 候选
        - verifier 验证
        - 最终 fault report
    """
    print("=" * 60)
    print("Starting INFERENCE mode")
    print("=" * 60)

    if args.issue_text:
        issue = args.issue_text
        sample_idx = None
    else:
        sample = load_swebench_sample(args.sample_idx, args.data_path)
        issue = sample.get("problem_statement", "")
        sample_idx = args.sample_idx

    repo_path = args.repo_path

    print(f"Issue preview: {issue[:150]}")
    graph, retriever = build_graph_and_retriever(repo_path)

    verifier = VerifierAgent(graph, use_llm=args.use_llm)
    reasoner = ReasonerAgent(graph, retriever, use_llm=args.use_llm)

    model = None
    if args.model_path:
        print(f"Loading model from {args.model_path}...")
        model = PPO.load(args.model_path)

    # 推理模式里不依赖真实 oracle bug_node，这里只是复用导航机制
    env = FaultLocalizationEnv(
        graph=graph,
        retriever=retriever,
        bug_query=issue,
        bug_node=args.bug_node if args.bug_node else None,
        max_steps=args.max_steps,
        top_k_retrieval=args.top_k_retrieval,
        query_mode=args.query_mode,
        reasoner=reasoner,
        verifier=verifier,
        seed=args.seed,
    )

    obs, info = env.reset()
    trace = []
    step_num = 0
    done = False

    print("\nStarting navigation...")

    # ========== 提前停止控制 ==========
    last_node_id = None
    consecutive_same_node = 0
    node_visit_counter = {}
    early_stop_reason = None

    # 可调阈值
    max_consecutive_same_node = 3   # 连续3次落到同一个节点就停
    max_total_visits_per_node = 5   # 同一节点累计访问5次就停
    # =================================

    print("\nStarting navigation...")
    
    # 动作名称映射
    action_names = ["JUMP", "CALL", "EXPAND", "SUBMIT"]
    
    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            # action 已经是 numpy 数组格式
        else:
            # 没有模型时的简单 fallback
            if step_num < max(args.max_steps - 1, 1):
                action = np.array([1, 0, 0, 0, 0])  # CALL
            else:
                action = np.array([3, 0, 0, 0, 0])  # SUBMIT
        
        # 获取动作类型（第一个元素）
        if isinstance(action, np.ndarray):
            action_type = int(action[0])
            action_details = action.tolist()
        else:
            action_type = int(action)
            action_details = action_type
        
        action_name = action_names[action_type] if action_type < len(action_names) else "UNKNOWN"
        
        prev_node_id = env.current_node
        prev_node = env.graph.nodes.get(prev_node_id) if prev_node_id else None
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        curr_node_id = env.current_node
        curr_node = env.graph.nodes.get(curr_node_id) if curr_node_id else None
        
        trace_item = {
            "step": step_num,
            "action": action_name,
            "action_details": action_details,
            "from_node_id": prev_node_id,
            "from_node_name": getattr(prev_node, "name", "unknown") if prev_node else "unknown",
            "to_node_id": curr_node_id,
            "to_node_name": getattr(curr_node, "name", "unknown") if curr_node else "unknown",
            "to_node_type": getattr(curr_node, "type", "unknown") if curr_node else "unknown",
            "file_path": getattr(curr_node, "file_path", "") if curr_node else "",
            "reward": float(reward),
        }
        trace.append(trace_item)
        
        print(
            f"  Step {step_num}: {action_name} -> "
            f"{trace_item['to_node_name']} ({trace_item['to_node_type']}) "
            f"reward={reward:.3f}"
        )
        
        step_num += 1
        if step_num >= args.max_steps:
            break

    # ----------------------------------------------------------
    # 生成 top-k 候选
    # ----------------------------------------------------------
    retrieval_results = retrieve_candidates(
        retriever=retriever,
        graph=graph,
        query=issue,
        top_k=max(args.top_k_retrieval, 10),
    )

    reranked = reasoner.rerank(issue, retrieval_results)

    candidate_reports = []
    for rank, (node_id, score) in enumerate(reranked[: args.final_top_k], start=1):
        verify_result = verifier.debate(issue, node_id)
        candidate_reports.append(
            serialize_candidate(
                graph=graph,
                node_id=node_id,
                score=score,
                rank=rank,
                verifier_result=verify_result,
            )
        )

    # 导航终点单独验证
    final_node_id = env.current_node
    final_node = graph.nodes.get(final_node_id)
    final_navigation_verification = verifier.debate(issue, final_node_id) if final_node_id is not None else None
    refiner = SnippetRefiner()
    snippet = []
    if final_node and final_node.code:
        snippet = refiner.refine(issue, final_node.code)
    # 最终主结果：优先取 rerank 后 top-1，而不是盲信 RL 最终停点
    primary_result = candidate_reports[0] if candidate_reports else {
        "rank": 1,
        "node_id": final_node_id,
        "score": 0.0,
        "entity_name": safe_attr(final_node, "name", "unknown") if final_node else "unknown",
        "entity_type": safe_attr(final_node, "type", "unknown") if final_node else "unknown",
        "file_path": safe_attr(final_node, "file_path", "") if final_node else "",
        "line_number": safe_attr(final_node, "line_number", None) if final_node else None,
        "verifier_verdict": final_navigation_verification.get("verdict") if final_navigation_verification else None,
        "evidence": final_navigation_verification.get("evidence", [])[:8] if final_navigation_verification else [],
        "code_preview": str(safe_attr(final_node, "code", "") or "")[:300] if final_node else "",
    }
    if isinstance(primary_result, dict):
        primary_result["snippet"] = snippet

    output = {
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "mode": "inference",
        "sample_idx": sample_idx,
        "repo_path": repo_path,
        "bug_description": issue[:1000],
        "primary_result": primary_result,
        "navigation_final_node": {
            "node_id": final_node_id,
            "entity_name": safe_attr(final_node, "name", "unknown") if final_node else "unknown",
            "entity_type": safe_attr(final_node, "type", "unknown") if final_node else "unknown",
            "file_path": safe_attr(final_node, "file_path", "") if final_node else "",
            "line_number": safe_attr(final_node, "line_number", None) if final_node else None,
            "verifier_result": final_navigation_verification,
        },
        "top_candidates": candidate_reports,
        "reasoner_trace": getattr(reasoner, "last_trace", None),
        "navigation_trace": {
            "total_steps": len(trace),
            "early_stop_reason": early_stop_reason,
            "path": trace,
        },
    }

    output_file = args.output_file or f"fl_result_{uuid.uuid4().hex[:8]}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("FAULT LOCALIZATION RESULT")
    print("=" * 60)
    print(f"Primary file: {primary_result.get('file_path', '')}")
    print(f"Primary entity: {primary_result.get('entity_name', '')} ({primary_result.get('entity_type', '')})")
    print(f"Primary score: {primary_result.get('score', 0.0):.4f}")
    print(f"Verifier verdict: {primary_result.get('verifier_verdict')}")
    print(f"Navigation steps: {len(trace)}")
    if early_stop_reason:
        print(f"Early stop: {early_stop_reason}")
    print(f"Saved to: {output_file}")
    print("=" * 60)

    return output



# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fault Localization System")

    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train")
    parser.add_argument("--repo_path", type=str, default="data_storage/repos/astropy")
    parser.add_argument("--data_path", type=str, default="data_storage/SWE-bench_Lite/data/test-00000-of-00001.parquet")

    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--issue_text", type=str, default=None)

    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--top_k_retrieval", type=int, default=5)
    parser.add_argument("--final_top_k", type=int, default=5)

    parser.add_argument("--model_output", type=str, default="best_model")
    parser.add_argument("--model_path", type=str, default="best_model.zip")
    parser.add_argument("--output_file", type=str, default=None)

    parser.add_argument("--bug_node", type=str, default=None)
    parser.add_argument("--query_mode", type=str, choices=["manual", "weak", "strong", "minimal"], default="weak")

    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # PPO params
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tensorboard_log", type=str, default="./logs")

    args = parser.parse_args()

    if args.mode == "train":
        train_mode(args)
    else:
        inference_mode(args)


if __name__ == "__main__":
    main()