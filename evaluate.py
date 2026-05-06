#!/usr/bin/env python3
"""
SWE-bench Lite 完整评估脚本
支持：
- Fault Localization Accuracy
- 效率对比
- 消融实验
- 混合检索 (BM25 + Dense)
"""

import json
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np

from graph.builder import GraphBuilder
from retrieval.bm25 import SimpleBM25
from retrieval.retriever import Retriever
from agent.rl_env import FaultLocalizationEnv
from verifier.debate_agent import VerifierAgent
from reasoner.reasoner_agent import ReasonerAgent
from stable_baselines3 import PPO


class SWEBenchEvaluator:
    """SWE-bench Lite 评估器"""
    
    def __init__(
        self,
        data_path: str,
        repo_path: str,
        split: str = "test", 
        model_path: Optional[str] = None,
        use_llm: bool = True,
        max_steps: int = 20,
        top_k_retrieval: int = 5,
        device: str = "cuda",
        # Dense Retrieval 参数
        use_dense: bool = False,
        dense_model_path: str = "./dense_retriever_model",
        retriever_alpha: float = 0.5,
        retriever_beta: float = 0.5,
    ):
        # 处理数据路径
        if split == "train":
            data_path = "data_storage/splits/train.parquet"
        elif split == "val":
            data_path = "data_storage/splits/val.parquet"
        else:
            data_path = "data_storage/splits/test.parquet"
        
        self.data_path = data_path
        self.repo_path = repo_path
        self.model_path = model_path
        self.use_llm = use_llm
        self.max_steps = max_steps
        self.top_k_retrieval = top_k_retrieval
        self.device = device
        
        # Dense Retrieval 配置
        self.use_dense = use_dense
        self.dense_model_path = dense_model_path
        self.retriever_alpha = retriever_alpha
        self.retriever_beta = retriever_beta
        
        # 加载数据
        self.df = pd.read_parquet(data_path)
        print(f"Loaded {len(self.df)} samples from {split} split")
        
        # 构建图和检索器
        self.graph, self.retriever = self._build_graph_and_retriever()
        
        # 加载模型（如果提供）
        self.model = None
        if model_path and Path(model_path).exists():
            self.model = PPO.load(model_path, device=device)
            print(f"Loaded model from {model_path}")
        elif model_path:
            print(f"Warning: Model file {model_path} not found")
        
    def _build_graph_and_retriever(self):
        """构建图和检索器（支持混合检索）"""
        print(f"Building graph from {self.repo_path}...")
        builder = GraphBuilder()
        graph = builder.build_from_repo(self.repo_path)
        
        corpus = {}
        for node_id, node in graph.nodes.items():
            name = getattr(node, "name", "") or ""
            doc = getattr(node, "doc", "") or ""
            file_path = getattr(node, "file_path", "") or ""
            code = getattr(node, "code", "") or ""
            corpus[node_id] = f"{name} {doc} {file_path} {str(code)[:300]}".strip()
        
        bm25 = SimpleBM25(corpus)
        
        # 创建混合检索器
        print(f"Creating retriever (use_dense={self.use_dense})")
        retriever = Retriever(
            bm25,
            alpha=self.retriever_alpha,
            beta=self.retriever_beta,
            use_dense=self.use_dense,
            dense_model_path=self.dense_model_path if self.use_dense else None,
            dense_device=self.device if self.use_dense else "cpu",
        )
        retriever.build_index(graph)
        
        return graph, retriever
    
    def _get_ground_truth(self, sample: Dict) -> List[str]:
        """从 SWE-bench 数据中提取 ground truth"""
        ground_truth = []
        
        if "patch" in sample and sample["patch"]:
            patch = sample["patch"]
            import re
            
            # 提取修改的文件路径
            file_matches = re.findall(r'diff --git a/(.+?) b/(.+?)\n', patch)
            for old_file, new_file in file_matches:
                file_path = new_file
                ground_truth.append(file_path)
                file_name = file_path.split('/')[-1].replace('.py', '')
                ground_truth.append(file_name)
            
            # 提取修改的函数名
            func_matches = re.findall(r'@@.*@@\s+def\s+(\w+)\s*\(', patch)
            for func in func_matches:
                ground_truth.append(func)
            
            # 提取修改的类名
            class_matches = re.findall(r'@@.*@@\s+class\s+(\w+)', patch)
            for cls in class_matches:
                ground_truth.append(cls)
        
        if "instance_id" in sample:
            instance_id = sample["instance_id"]
            if "__" in instance_id:
                parts = instance_id.split("__")
                if len(parts) >= 2:
                    ground_truth.append(parts[1])
        
        return list(set(ground_truth))
    
    def _is_correct(self, prediction: str, ground_truth: List[str]) -> bool:
        """判断预测是否正确"""
        if not ground_truth or not prediction:
            return False
        
        prediction_lower = prediction.lower()
        
        for gt in ground_truth:
            gt_lower = gt.lower()
            
            if prediction == gt:
                return True
            
            if gt_lower in prediction_lower or prediction_lower in gt_lower:
                return True
            
            pred_name = prediction_lower.split('/')[-1].split('.')[0]
            gt_name = gt_lower.split('/')[-1].split('.')[0]
            if pred_name == gt_name:
                return True
        
        return False
    
    def evaluate_single_sample(
        self, 
        sample: Dict, 
        use_rl: bool = True
    ) -> Dict:
        """评估单个样本"""
        issue = sample.get("problem_statement", "")
        ground_truth = self._get_ground_truth(sample)
        
        verifier = VerifierAgent(self.graph, use_llm=self.use_llm)
        reasoner = ReasonerAgent(self.graph, self.retriever, use_llm=self.use_llm)
        
        env = FaultLocalizationEnv(
            graph=self.graph,
            retriever=self.retriever,
            bug_query=issue,
            max_steps=self.max_steps,
            top_k_retrieval=self.top_k_retrieval,
            reasoner=reasoner,
            verifier=verifier,
        )
        
        start_time = time.time()
        obs, _ = env.reset()
        
        trajectory = []
        step_count = 0
        actions_taken = {"JUMP": 0, "CALL": 0, "EXPAND": 0, "SUBMIT": 0}
        
        for step in range(self.max_steps):
            if use_rl and self.model:
                action, _ = self.model.predict(obs, deterministic=True)
            else:
                # 基线：启发式策略 - 先检索，最后提交
                if step < self.max_steps - 1:
                    action = np.array([1, 0, 0, 0, 0])  # CALL
                else:
                    action = np.array([3, 0, 0, 0, 0])  # SUBMIT
            
            action_type = action[0] if hasattr(action, '__getitem__') else action
            action_name = ["JUMP", "CALL", "EXPAND", "SUBMIT"][action_type]
            actions_taken[action_name] += 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            trajectory.append({
                "step": step,
                "action": action_name,
                "node": info.get("current_node_name", ""),
                "reward": reward,
            })
            
            if terminated or truncated:
                break
        
        elapsed_time = time.time() - start_time
        
        final_node = env.current_node
        final_node_name = getattr(self.graph.nodes.get(final_node), "name", "") if final_node else ""
        
        is_correct = self._is_correct(final_node_name, ground_truth)
        
        return {
            "sample_idx": sample.get("instance_id", "unknown"),
            "issue": issue[:200],
            "ground_truth": ground_truth,
            "prediction": final_node_name,
            "is_correct": is_correct,
            "steps": step_count,
            "time": elapsed_time,
            "actions": actions_taken,
            "trajectory": trajectory,
            "final_reward": reward,
        }
    
    def evaluate_all(
        self, 
        sample_indices: Optional[List[int]] = None,
        use_rl: bool = True
    ) -> pd.DataFrame:
        """评估所有样本"""
        if sample_indices is None:
            sample_indices = list(range(len(self.df)))
        
        results = []
        for idx in sample_indices:
            print(f"\nEvaluating sample {idx+1}/{len(sample_indices)}...")
            sample = self.df.iloc[idx].to_dict()
            result = self.evaluate_single_sample(sample, use_rl=use_rl)
            results.append(result)
            
            status = "✅" if result["is_correct"] else "❌"
            print(f"  {status} Predicted: {result['prediction'][:50]}, Steps: {result['steps']}")
        
        return pd.DataFrame(results)
    
    def run_accuracy_experiment(self) -> Dict:
        """运行准确率实验"""
        print("\n" + "=" * 60)
        print("EXPERIMENT 1: Fault Localization Accuracy")
        print("=" * 60)
        
        configs = [
            {"name": "Baseline (Fixed Strategy)", "use_rl": False},
            {"name": "RL (Ours)", "use_rl": True},
        ]
        
        results = {}
        for config in configs:
            print(f"\n--- {config['name']} ---")
            df = self.evaluate_all(use_rl=config["use_rl"])
            
            metrics = {
                "accuracy": df["is_correct"].mean(),
                "avg_steps": df["steps"].mean(),
                "avg_call_count": df["actions"].apply(lambda x: x["CALL"]).mean(),
                "avg_expand_count": df["actions"].apply(lambda x: x["EXPAND"]).mean(),
                "total_samples": len(df),
            }
            results[config["name"]] = metrics
            
            print(f"  Accuracy: {metrics['accuracy']:.2%}")
            print(f"  Avg Steps: {metrics['avg_steps']:.1f}")
            print(f"  Avg CALL: {metrics['avg_call_count']:.1f}")
            print(f"  Avg EXPAND: {metrics['avg_expand_count']:.1f}")
        
        return results
    
    def run_efficiency_experiment(self) -> Dict:
        """运行效率对比实验"""
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: Efficiency Comparison")
        print("=" * 60)
        
        step_configs = [5, 10, 15, 20, 30]
        results = {}
        
        for max_steps in step_configs:
            print(f"\n--- Max Steps: {max_steps} ---")
            self.max_steps = max_steps
            
            df = self.evaluate_all(use_rl=True)
            
            metrics = {
                "avg_steps_used": df["steps"].mean(),
                "avg_time": df["time"].mean(),
                "accuracy": df["is_correct"].mean(),
                "success_rate": (df["steps"] < max_steps).mean(),
            }
            results[max_steps] = metrics
            
            print(f"  Avg steps used: {metrics['avg_steps_used']:.1f}/{max_steps}")
            print(f"  Accuracy: {metrics['accuracy']:.2%}")
            print(f"  Time: {metrics['avg_time']:.2f}s")
        
        return results
    
    def save_results(self, results: Dict, filename: str):
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filename}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data_storage/SWE-bench_Lite/data/test-00000-of-00001.parquet")
    parser.add_argument("--repo_path", type=str, default="data_storage/repos/astropy")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--experiment", type=str, choices=["accuracy", "efficiency", "all"], default="accuracy")
    parser.add_argument("--max_steps", type=int, default=20, help="Maximum steps per episode")
    
    # Dense Retrieval 参数
    parser.add_argument("--use_dense", action="store_true")
    parser.add_argument("--dense_model_path", type=str, default="./dense_retriever_model")
    parser.add_argument("--retriever_alpha", type=float, default=0.5)
    parser.add_argument("--retriever_beta", type=float, default=0.5)
    
    args = parser.parse_args()
    
    evaluator = SWEBenchEvaluator(
        data_path=args.data_path,
        repo_path=args.repo_path,
        split=args.split,
        model_path=args.model_path,
        use_llm=args.use_llm,
        device=args.device,
        max_steps=args.max_steps,
        use_dense=args.use_dense,
        dense_model_path=args.dense_model_path,
        retriever_alpha=args.retriever_alpha,
        retriever_beta=args.retriever_beta,
    )
    
    all_results = {}
    
    if args.experiment in ["accuracy", "all"]:
        results = evaluator.run_accuracy_experiment()
        all_results["accuracy"] = results
        evaluator.save_results(results, "results_accuracy.json")
    
    if args.experiment in ["efficiency", "all"]:
        results = evaluator.run_efficiency_experiment()
        all_results["efficiency"] = results
        evaluator.save_results(results, "results_efficiency.json")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for exp_name, results in all_results.items():
        print(f"\n{exp_name.upper()}:")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()