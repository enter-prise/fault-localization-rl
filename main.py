import argparse
import json
import uuid
import time
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


def build_graph_and_retriever(
    repo_path: str,
    use_dense: bool = False,
    dense_model_path: str = "./dense_retriever_model",
    alpha: float = 0.5,
    beta: float = 0.5,
):
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
    
    # 创建检索器（BM25 + Dense，可选）
    retriever = Retriever(
        bm25,
        alpha=alpha,
        beta=beta,
        use_dense=use_dense,
        dense_model_path=dense_model_path if use_dense else None,
    )

    try:
        retriever.build_index(graph)
    except Exception as e:
        print(f"Warning: Failed to build index: {e}")

    return graph, retriever


def retrieve_candidates(retriever, graph, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
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

    if args.split == "train":
        data_path = "data_storage/splits/train.parquet"
    elif args.split == "val":
        data_path = "data_storage/splits/val.parquet"
    else:
        data_path = args.data_path

    print(f"Loading {args.split} split from {data_path}")
    sample = load_swebench_sample(args.sample_idx, data_path)
    issue = sample.get("problem_statement", "")
    repo_path = args.repo_path

    print(f"Training sample_idx={args.sample_idx}")
    print(f"Issue preview: {issue[:120]}")

    graph, retriever = build_graph_and_retriever(
        repo_path=repo_path,
        use_dense=args.use_dense,
        dense_model_path=args.dense_model_path,
        alpha=args.retriever_alpha,
        beta=args.retriever_beta,
    )


    reasoner = ReasonerAgent(graph, retriever, model_name=args.llm_model, use_llm=args.use_llm)
    verifier = VerifierAgent(graph, model_name=args.llm_model, use_llm=args.use_llm)

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
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    print("Starting PPO training...")
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        device=args.device,
        tensorboard_log=args.tensorboard_log,
        policy_kwargs=dict(
            net_arch=[128, 128],
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
    start_time = time.time()
    """
    推理模式
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

    graph, retriever = build_graph_and_retriever(
        repo_path=repo_path,
        use_dense=args.use_dense,
        dense_model_path=args.dense_model_path,
        alpha=args.retriever_alpha,
        beta=args.retriever_beta,
    )

    verifier = VerifierAgent(graph, model_name=args.llm_model, use_llm=args.use_llm)
    reasoner = ReasonerAgent(graph, retriever, model_name=args.llm_model, use_llm=args.use_llm)

    model = None
    if args.model_path:
        print(f"Loading model from {args.model_path}...")
        model = PPO.load(args.model_path)

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
        auto_terminate_on_exact_hit=False,
    )

    obs, info = env.reset()
    trace = []
    step_num = 0
    done = False

    print("\nStarting navigation...")

# 注释掉整个早停控制块
# ========== 提前停止控制 ==========
# last_node_id = None
# consecutive_same_node = 0
# node_visit_counter = {}
# early_stop_reason = None
# max_consecutive_same_node = 3
# max_total_visits_per_node = 5
# =================================
    
    action_names = ["JUMP", "CALL", "EXPAND", "SUBMIT"]
    
    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            if step_num < max(args.max_steps - 1, 1):
                action = np.array([1, 0, 0, 0, 0])
            else:
                action = np.array([3, 0, 0, 0, 0])
        
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
        
        print(f"  Step {step_num}: {action_name} -> {trace_item['to_node_name']} ({trace_item['to_node_type']}) reward={reward:.3f}")
        
        step_num += 1
        if step_num >= args.max_steps:
            break

    # 生成 top-k 候选
    retrieval_results = retrieve_candidates(
        retriever=retriever,
        graph=graph,
        query=issue,
        top_k=max(args.top_k_retrieval, 10),
    )

    rl_candidates = []
    seen_entity_ids = set()
    rl_candidate_sources = {}

    def add_rl_candidate(entity_id, score=0.0, source="candidate_pool"):
        if entity_id is None:
            return
        entity_id = str(entity_id)
        if entity_id not in graph.nodes:
            return
        if entity_id in seen_entity_ids:
            existing_source = rl_candidate_sources.get(entity_id, "")
            if source and source not in existing_source.split("+"):
                rl_candidate_sources[entity_id] = f"{existing_source}+{source}" if existing_source else source
            return
        seen_entity_ids.add(entity_id)
        rl_candidate_sources[entity_id] = source
        rl_candidates.append((entity_id, score))

    for candidate in getattr(env, "last_submit_candidates", []):
        entity_id = candidate.get("entity_id")
        score = candidate.get("relevance_score", candidate.get("score", 0.0))
        add_rl_candidate(entity_id, score, source="submit_candidates")

    current_score = 0.5
    for candidate in getattr(env, "candidate_pool", []):
        if str(candidate.get("entity_id")) == str(env.current_node):
            current_score = candidate.get("relevance_score", candidate.get("score", current_score))
            break
    add_rl_candidate(env.current_node, current_score, source="current_node")

    for candidate in getattr(env, "candidate_pool", []):
        entity_id = candidate.get("entity_id")
        score = candidate.get("relevance_score", candidate.get("score", 0.0))
        add_rl_candidate(entity_id, score, source="candidate_pool")

    candidate_pool_empty = not bool(getattr(env, "candidate_pool", []))
    only_current_node_candidate = (
        len(rl_candidates) == 1
        and str(rl_candidates[0][0]) == str(env.current_node)
    )
    retrieval_fallback_used = not bool(rl_candidates) or (candidate_pool_empty and only_current_node_candidate)
    rerank_input_candidates = rl_candidates if rl_candidates else retrieval_results
    reranked = reasoner.rerank(issue, rerank_input_candidates)

    candidate_reports = []
    for rank, (node_id, score) in enumerate(reranked[: args.final_top_k], start=1):
        verify_result = verifier.debate(issue, node_id)
        
        node = graph.nodes.get(node_id)
        node_name = safe_attr(node, "name", "unknown")
        node_type = safe_attr(node, "type", "unknown")
        
        line_start = getattr(node, "lineno", None) or getattr(node, "line_number", None)
        line_end = getattr(node, "end_lineno", None) or getattr(node, "end_line_number", None) or line_start
                
        class_name = None
        function_name = None
        if node_type == "method":
            function_name = node_name
            for neighbor in graph.get_neighbors(node_id, edge_type="contains"):
                neighbor_node = graph.nodes.get(neighbor)
                if neighbor_node and safe_attr(neighbor_node, "type", "") == "class":
                    class_name = safe_attr(neighbor_node, "name", "")
                    break
        elif node_type == "function":
            function_name = node_name
        elif node_type == "class":
            class_name = node_name
        
        retrieval_source = ["semantic"]
        if hasattr(retriever, '_bm25_search'):
            retrieval_source.append("bm25")
        if hasattr(reasoner, 'last_trace') and reasoner.last_trace:
            if reasoner.last_trace.get("mode") == "llm_hybrid":
                retrieval_source.append("reranked")
        
        candidate_reports.append({
            "rank": rank,
            "entity_id": node_id,
            "entity_name": node_name,
            "entity_type": node_type,
            "file_path": safe_attr(node, "file_path", ""),
            "line_range": [line_start, line_end] if line_start else None,
            "class_name": class_name,
            "function_name": function_name,
            "code_snippet": str(safe_attr(node, "code", "") or "")[:500],
            "retrieval_source": retrieval_source,
            "relevance_score": round(score, 4),
            "policy_score": round(score, 4),
            "verification_status": verify_result.get("verdict", "unknown").capitalize(),
            "confidence_score": verify_result.get("confidence", 0.5),
            "reasoning": verify_result.get("evidence", ["No reasoning provided"])[0] if verify_result.get("evidence") else "No reasoning provided",
        })

    # 导航终点单独验证
    final_node_id = env.current_node
    final_node = graph.nodes.get(final_node_id) if final_node_id else None
    final_navigation_verification = verifier.debate(issue, final_node_id) if final_node_id is not None else None
    refiner = SnippetRefiner()
    snippet = []
    if final_node and hasattr(final_node, "code") and final_node.code:
        snippet = refiner.refine(issue, final_node.code)
    
    primary_result = candidate_reports[0] if candidate_reports else {
        "rank": 1,
        "entity_id": final_node_id,
        "entity_name": safe_attr(final_node, "name", "unknown") if final_node else "unknown",
        "entity_type": safe_attr(final_node, "type", "unknown") if final_node else "unknown",
        "file_path": safe_attr(final_node, "file_path", "") if final_node else "",
        "line_range": None,
        "class_name": None,
        "function_name": safe_attr(final_node, "name", "unknown") if final_node else "unknown",
        "code_snippet": str(safe_attr(final_node, "code", "") or "")[:500] if final_node else "",
        "retrieval_source": ["navigation"],
        "relevance_score": 0.0,
        "policy_score": 0.0,
        "verification_status": final_navigation_verification.get("verdict", "unknown").capitalize() if final_navigation_verification else "Unknown",
        "confidence_score": final_navigation_verification.get("confidence", 0.0) if final_navigation_verification else 0.0,
        "reasoning": final_navigation_verification.get("evidence", ["No reasoning"])[0] if final_navigation_verification and final_navigation_verification.get("evidence") else "Navigation end point",
    }
    
    if isinstance(primary_result, dict) and snippet:
        primary_result["snippet"] = snippet

    # 构建导航轨迹
    navigation_trajectory = []
    for step in trace:
        traj_item = {
            "step": step["step"] + 1,
            "current_entity": step["to_node_name"],
            "action": step["action"],
            "observation": f"Moved from {step['from_node_name']} to {step['to_node_name']}, reward={step['reward']:.3f}",
        }
        if step["action"] == "CALL" and "action_details" in step:
            tool_names = ["SemanticScout", "CodeExplorer", "ContextProbe"]
            tool_idx = step["action_details"][2] if isinstance(step["action_details"], list) and len(step["action_details"]) > 2 else 0
            traj_item["tool"] = tool_names[tool_idx] if tool_idx < len(tool_names) else "Unknown"
        navigation_trajectory.append(traj_item)

    # Verifier 反馈
    verifier_feedback = {
        "claim": f"The fault is most likely in {primary_result.get('entity_name', 'unknown')}",
        "challenge": "Could be class-level misconfiguration instead of method-level bug" if primary_result.get("entity_type") == "method" else "Could be broader context issue",
        "final_decision": primary_result.get("verification_status", "Unknown"),
        "confidence_score": primary_result.get("confidence_score", 0.5),
    }

    # 运行元数据
    run_metadata = {
        "total_steps": len(trace),
        "tool_calls": sum(1 for t in trace if t["action"] == "CALL"),
        "expanded_nodes": sum(1 for t in trace if t["action"] == "EXPAND"),
        "runtime_seconds": round(time.time() - start_time, 2),
        "model_name": "RL-NavFL",
        "policy_model": "PPO",
    }

    output = {
        "issue_id": f"custom-{uuid.uuid4().hex[:8]}",
        "repo_name": repo_path.split("/")[-1],
        "query": issue[:500],
        "final_prediction": primary_result,
        "rl_candidates": [
            {
                "entity_id": node_id,
                "score": round(float(score), 4),
                "source": rl_candidate_sources.get(str(node_id), "unknown"),
            }
            for node_id, score in rl_candidates
        ],
        "retrieval_fallback_used": retrieval_fallback_used,
        "top_k_candidates": candidate_reports,
        "navigation_trajectory": navigation_trajectory,
        "verifier_feedback": verifier_feedback,
        "run_metadata": run_metadata,
    }

    output_file = args.output_file or f"fl_result_{uuid.uuid4().hex[:8]}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("FAULT LOCALIZATION RESULT")
    print("=" * 60)
    print(f"Primary file: {primary_result.get('file_path', '')}")
    print(f"Primary entity: {primary_result.get('entity_name', '')} ({primary_result.get('entity_type', '')})")
    print(f"Confidence: {primary_result.get('confidence_score', 0.0):.2f}")
    print(f"Verifier verdict: {primary_result.get('verification_status', '')}")
    print(f"Navigation steps: {len(trace)}")
    print(f"Runtime: {run_metadata['runtime_seconds']:.2f} seconds")
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
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"],
                        help="Data split to use")

    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--issue_text", type=str, default=None)

    # Dense Retrieval 参数
    parser.add_argument("--use_dense", action="store_true", 
                        help="Enable dense retrieval")
    parser.add_argument("--dense_model_path", type=str, default="./dense_retriever_model",
                        help="Path to dense retriever model")
    parser.add_argument("--retriever_alpha", type=float, default=0.5,
                        help="BM25 weight")
    parser.add_argument("--retriever_beta", type=float, default=0.5,
                        help="Dense retrieval weight")

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
    parser.add_argument("--llm_model", type=str, default="qwen2.5-coder:32b",
                    help="LLM model to use (e.g., llama3:8b, qwen2.5-coder:32b)")


    # PPO params
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tensorboard_log", type=str, default="./logs")
    
    # PPO 参数
    parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps per rollout")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs per rollout")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

    args = parser.parse_args()

    if args.mode == "train":
        train_mode(args)
    else:
        inference_mode(args)


if __name__ == "__main__":
    main()
