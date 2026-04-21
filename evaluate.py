import argparse
import csv
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from stable_baselines3 import PPO

from main import (
    safe_attr,
    build_graph_and_retriever,
    retrieve_candidates,
    serialize_candidate,
)
from verifier.debate_agent import VerifierAgent
from reasoner.reasoner_agent import ReasonerAgent
from agent.rl_env import FaultLocalizationEnv


# =========================================================
# Utility
# =========================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_path(p: str) -> str:
    if not p:
        return ""
    return p.replace("\\", "/").strip()


def safe_json_dump(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# =========================================================
# Oracle extraction
# =========================================================

def extract_file_paths_from_patch(patch_text: str) -> List[str]:
    """
    从 unified diff / patch 中提取文件路径。
    支持类似：
      diff --git a/path/to/file.py b/path/to/file.py
      --- a/path/to/file.py
      +++ b/path/to/file.py
    """
    if not patch_text:
        return []

    paths = set()

    patterns = [
        r"diff --git a/(.*?) b/(.*?)\n",
        r"\+\+\+ b/(.*?)\n",
        r"--- a/(.*?)\n",
    ]

    for pat in patterns:
        for m in re.finditer(pat, patch_text):
            groups = [g for g in m.groups() if g]
            for g in groups:
                g = normalize_path(g)
                if g and g != "/dev/null":
                    paths.add(g)

    py_paths = [p for p in paths if p.endswith(".py")]
    if py_paths:
        return sorted(py_paths)

    return sorted(paths)


def extract_oracle_from_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    尽量从数据集中抽取 file-level oracle。
    """
    patch_candidates = [
        sample.get("patch"),
        sample.get("gold_patch"),
        sample.get("solution"),
    ]

    patch_text = None
    for p in patch_candidates:
        if isinstance(p, str) and p.strip():
            patch_text = p
            break

    oracle_files = extract_file_paths_from_patch(patch_text or "")

    return {
        "oracle_files": oracle_files,
        "has_file_oracle": len(oracle_files) > 0,
    }


# =========================================================
# Repo resolution
# =========================================================

def infer_repo_name_from_oracle_files(oracle_files: List[str]) -> Optional[str]:
    """
    从 oracle file 路径推断仓库名。
    例如：
      astropy/modeling/separable.py -> astropy
      django/forms/widgets.py -> django
    """
    if not oracle_files:
        return None

    prefixes = []
    for f in oracle_files:
        f = normalize_path(f)
        if not f:
            continue
        parts = f.split("/")
        if len(parts) >= 2:
            prefixes.append(parts[0])

    if not prefixes:
        return None

    # 多数投票
    counts = {}
    for p in prefixes:
        counts[p] = counts.get(p, 0) + 1

    return sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]


def resolve_repo_path(sample: Dict[str, Any], repos_root: str) -> Dict[str, Any]:
    """
    根据 sample 自动解析仓库路径。
    """
    oracle = extract_oracle_from_sample(sample)
    oracle_files = oracle["oracle_files"]

    repo_name = infer_repo_name_from_oracle_files(oracle_files)
    if repo_name is None:
        return {
            "repo_name": None,
            "repo_path": None,
            "exists": False,
            "oracle_files": oracle_files,
            "has_file_oracle": oracle["has_file_oracle"],
        }

    repo_path = os.path.join(repos_root, repo_name)

    return {
        "repo_name": repo_name,
        "repo_path": repo_path,
        "exists": os.path.isdir(repo_path),
        "oracle_files": oracle_files,
        "has_file_oracle": oracle["has_file_oracle"],
    }


# =========================================================
# Single-sample inference for evaluation
# =========================================================

def run_single_inference(
    issue: str,
    repo_path: str,
    model,
    args,
) -> Dict[str, Any]:
    """
    不调用 main.py 的 inference_mode，避免写很多中间文件。
    这里直接执行同样的推理逻辑，并返回结构化结果。
    """
    graph, retriever = build_graph_and_retriever(repo_path)

    verifier = VerifierAgent(graph, use_llm=args.use_llm)
    reasoner = ReasonerAgent(graph, retriever, use_llm=args.use_llm)

    env = FaultLocalizationEnv(
        graph=graph,
        retriever=retriever,
        bug_query=issue,
        bug_node=args.bug_node if getattr(args, "bug_node", None) else None,
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

    last_node_id = None
    consecutive_same_node = 0
    node_visit_counter = {}
    early_stop_reason = None

    max_consecutive_same_node = args.max_consecutive_same_node
    max_total_visits_per_node = args.max_total_visits_per_node
    min_steps_before_stop = args.min_steps_before_stop

    start_time = time.time()

    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            if step_num < max(args.max_steps - 1, 1):
                action = 1
            else:
                action = 2

        action_name = ["JUMP", "RETRIEVE", "SUBMIT"][action]

        prev_node_id = env.current_node
        prev_node = env.graph.nodes.get(prev_node_id)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        curr_node_id = env.current_node
        curr_node = env.graph.nodes.get(curr_node_id)

        trace_item = {
            "step": step_num,
            "action": action_name,
            "from_node_id": prev_node_id,
            "from_node_name": safe_attr(prev_node, "name", "unknown") if prev_node else "unknown",
            "to_node_id": curr_node_id,
            "to_node_name": safe_attr(curr_node, "name", "unknown") if curr_node else "unknown",
            "to_node_type": safe_attr(curr_node, "type", "unknown") if curr_node else "unknown",
            "file_path": safe_attr(curr_node, "file_path", "") if curr_node else "",
            "reward": float(reward),
        }
        trace.append(trace_item)

        # 高置信 top-1 提前停止（至少走够一定步数）
        live_retrieval_results = retrieve_candidates(
            retriever=retriever,
            graph=graph,
            query=issue,
            top_k=max(args.top_k_retrieval, 10),
        )
        live_reranked = reasoner.rerank(issue, live_retrieval_results)

        live_verifier_result = None
        if curr_node_id is not None:
            live_verifier_result = verifier.debate(issue, curr_node_id)

        if (
            step_num >= min_steps_before_stop
            and curr_node_id is not None
            and live_reranked
            and curr_node_id == live_reranked[0][0]
            and live_verifier_result is not None
            and live_verifier_result.get("verdict", False)
        ):
            early_stop_reason = (
                f"early stop: high confidence top-1 match "
                f"(node_id={curr_node_id})"
            )
            break

        # 重复节点停止
        if curr_node_id == last_node_id:
            consecutive_same_node += 1
        else:
            consecutive_same_node = 1
            last_node_id = curr_node_id

        node_visit_counter[curr_node_id] = node_visit_counter.get(curr_node_id, 0) + 1

        if consecutive_same_node >= max_consecutive_same_node:
            early_stop_reason = (
                f"early stop: same node repeated consecutively "
                f"{consecutive_same_node} times (node_id={curr_node_id})"
            )
            break

        if node_visit_counter[curr_node_id] >= max_total_visits_per_node:
            early_stop_reason = (
                f"early stop: node visited too many times "
                f"({node_visit_counter[curr_node_id]} visits, node_id={curr_node_id})"
            )
            break

        step_num += 1
        if step_num >= args.max_steps:
            break

    runtime_sec = time.time() - start_time

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

    final_node_id = env.current_node
    final_node = graph.nodes.get(final_node_id)
    final_navigation_verification = verifier.debate(issue, final_node_id) if final_node_id is not None else None

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

    return {
        "primary_result": primary_result,
        "top_candidates": candidate_reports,
        "reasoner_trace": getattr(reasoner, "last_trace", None),
        "navigation_trace": {
            "total_steps": len(trace),
            "early_stop_reason": early_stop_reason,
            "path": trace,
        },
        "runtime_sec": runtime_sec,
    }


# =========================================================
# Metrics
# =========================================================

def compute_file_level_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "n_samples": 0,
            "file_oracle_coverage": 0,
            "top1_file_acc": None,
            "top3_file_acc": None,
            "top5_file_acc": None,
            "avg_steps": None,
            "avg_runtime_sec": None,
        }

    file_oracle_results = [r for r in results if r.get("has_file_oracle", False)]
    covered = len(file_oracle_results)

    top1_hits = 0
    top3_hits = 0
    top5_hits = 0

    for r in file_oracle_results:
        oracle_files = set(normalize_path(x) for x in r.get("oracle_files", []))
        pred_files = [normalize_path(c.get("file_path", "")) for c in r.get("top_candidates", [])]

        if len(pred_files) >= 1 and pred_files[0] in oracle_files:
            top1_hits += 1
        if any(p in oracle_files for p in pred_files[:3]):
            top3_hits += 1
        if any(p in oracle_files for p in pred_files[:5]):
            top5_hits += 1

    avg_steps = sum(r.get("navigation_steps", 0) for r in results) / total
    avg_runtime_sec = sum(r.get("runtime_sec", 0.0) for r in results) / total

    return {
        "n_samples": total,
        "file_oracle_coverage": covered,
        "top1_file_acc": (top1_hits / covered) if covered > 0 else None,
        "top3_file_acc": (top3_hits / covered) if covered > 0 else None,
        "top5_file_acc": (top5_hits / covered) if covered > 0 else None,
        "avg_steps": avg_steps,
        "avg_runtime_sec": avg_runtime_sec,
    }


# =========================================================
# Main evaluation loop
# =========================================================

def evaluate(args):
    ensure_dir(args.output_dir)

    model = None
    if args.model_path:
        print(f"Loading model from {args.model_path} ...")
        model = PPO.load(args.model_path)

    df = pd.read_parquet(args.data_path)
    total_rows = len(df)

    start_idx = args.start_idx
    end_idx = min(args.end_idx if args.end_idx is not None else total_rows, total_rows)

    print("=" * 60)
    print("BATCH EVALUATION")
    print("=" * 60)
    print(f"Dataset rows: {total_rows}")
    print(f"Evaluating range: [{start_idx}, {end_idx})")
    print(f"Repos root: {args.repos_root}")
    print("=" * 60)

    results = []
    skipped_no_oracle = 0
    skipped_repo_not_found = 0

    for idx in range(start_idx, end_idx):
        sample = dict(df.iloc[idx])
        issue = sample.get("problem_statement", "")
        repo_info = resolve_repo_path(sample, args.repos_root)

        if not repo_info["has_file_oracle"]:
            skipped_no_oracle += 1
            print(f"\n[{idx}] Skipped (no file-level oracle)")
            continue

        if not repo_info["exists"]:
            skipped_repo_not_found += 1
            print(f"\n[{idx}] Skipped (repo not found: {repo_info['repo_name']})")
            continue

        repo_name = repo_info["repo_name"]
        repo_path = repo_info["repo_path"]
        oracle_files = repo_info["oracle_files"]

        print(f"\n[{idx}] Running inference ... repo={repo_name}")
        try:
            inference_output = run_single_inference(
                issue=issue,
                repo_path=repo_path,
                model=model,
                args=args,
            )

            record = {
                "sample_idx": idx,
                "resolved_repo_name": repo_name,
                "resolved_repo_path": repo_path,
                "issue_preview": issue[:150],
                "oracle_files": oracle_files,
                "has_file_oracle": repo_info["has_file_oracle"],
                "primary_file": inference_output["primary_result"].get("file_path", ""),
                "primary_entity": inference_output["primary_result"].get("entity_name", ""),
                "primary_score": inference_output["primary_result"].get("score", 0.0),
                "navigation_steps": inference_output["navigation_trace"].get("total_steps", 0),
                "early_stop_reason": inference_output["navigation_trace"].get("early_stop_reason"),
                "runtime_sec": inference_output.get("runtime_sec", 0.0),
                "top_candidates": inference_output["top_candidates"],
                "primary_result": inference_output["primary_result"],
                "navigation_trace": inference_output["navigation_trace"],
                "reasoner_trace": inference_output["reasoner_trace"],
            }

            results.append(record)

            print(f"  Primary file: {record['primary_file']}")
            print(f"  Primary entity: {record['primary_entity']}")
            print(f"  Steps: {record['navigation_steps']}")
            print(f"  Runtime: {record['runtime_sec']:.3f}s")

        except Exception as e:
            print(f"  Failed on sample {idx}: {e}")
            results.append({
                "sample_idx": idx,
                "resolved_repo_name": repo_name,
                "resolved_repo_path": repo_path,
                "issue_preview": issue[:150],
                "oracle_files": oracle_files,
                "has_file_oracle": repo_info["has_file_oracle"],
                "error": str(e),
                "navigation_steps": 0,
                "runtime_sec": 0.0,
                "top_candidates": [],
                "primary_result": {},
                "navigation_trace": {},
                "reasoner_trace": {},
            })

    metrics = compute_file_level_metrics(results)
    metrics["repos_root"] = args.repos_root
    metrics["min_steps_before_stop"] = args.min_steps_before_stop
    metrics["skipped_no_oracle"] = skipped_no_oracle
    metrics["skipped_repo_not_found"] = skipped_repo_not_found

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(args.output_dir, f"eval_summary_{timestamp}.json")
    details_json_path = os.path.join(args.output_dir, f"eval_details_{timestamp}.json")
    details_csv_path = os.path.join(args.output_dir, f"eval_details_{timestamp}.csv")

    safe_json_dump(metrics, summary_path)
    safe_json_dump(results, details_json_path)

    with open(details_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_idx",
                "resolved_repo_name",
                "resolved_repo_path",
                "issue_preview",
                "has_file_oracle",
                "oracle_files",
                "primary_file",
                "primary_entity",
                "primary_score",
                "navigation_steps",
                "early_stop_reason",
                "runtime_sec",
                "error",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "sample_idx": r.get("sample_idx"),
                "resolved_repo_name": r.get("resolved_repo_name", ""),
                "resolved_repo_path": r.get("resolved_repo_path", ""),
                "issue_preview": r.get("issue_preview"),
                "has_file_oracle": r.get("has_file_oracle"),
                "oracle_files": "|".join(r.get("oracle_files", [])),
                "primary_file": r.get("primary_file", ""),
                "primary_entity": r.get("primary_entity", ""),
                "primary_score": r.get("primary_score", ""),
                "navigation_steps": r.get("navigation_steps", 0),
                "early_stop_reason": r.get("early_stop_reason", ""),
                "runtime_sec": r.get("runtime_sec", 0.0),
                "error": r.get("error", ""),
            })

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Summary JSON: {summary_path}")
    print(f"Details JSON: {details_json_path}")
    print(f"Details CSV : {details_csv_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation for RL-NavFL")

    # 注意：这里不再要求固定 repo_path，而是要求 repos_root
    parser.add_argument("--repos_root", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="eval_outputs")

    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=10)

    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--top_k_retrieval", type=int, default=5)
    parser.add_argument("--final_top_k", type=int, default=5)
    parser.add_argument("--query_mode", type=str, choices=["manual", "weak", "strong", "minimal"], default="weak")
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bug_node", type=str, default=None)

    parser.add_argument("--max_consecutive_same_node", type=int, default=3)
    parser.add_argument("--max_total_visits_per_node", type=int, default=5)
    parser.add_argument("--min_steps_before_stop", type=int, default=3)

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()