#!/usr/bin/env python3
"""对比 RL 策略 vs 固定策略"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from graph.builder import GraphBuilder
from retrieval.bm25 import SimpleBM25
from retrieval.retriever import Retriever
from agent.rl_env import FaultLocalizationEnv
from verifier.debate_agent import VerifierAgent
from reasoner.reasoner_agent import ReasonerAgent
from stable_baselines3 import PPO


def evaluate_with_fixed_strategy(evaluator, sample_indices):
    """使用固定策略：先检索3次，再扩展3次，最后提交"""
    results = []
    for idx in tqdm(sample_indices, desc="Fixed Strategy"):
        sample = evaluator.df.iloc[idx].to_dict()
        issue = sample.get("problem_statement", "")
        
        # 创建环境
        verifier = VerifierAgent(evaluator.graph, use_llm=evaluator.use_llm)
        reasoner = ReasonerAgent(evaluator.graph, evaluator.retriever, use_llm=evaluator.use_llm)
        
        env = FaultLocalizationEnv(
            graph=evaluator.graph,
            retriever=evaluator.retriever,
            bug_query=issue,
            max_steps=20,
            top_k_retrieval=5,
            reasoner=reasoner,
            verifier=verifier,
        )
        
        obs, _ = env.reset()
        step_count = 0
        actions_taken = {"JUMP": 0, "CALL": 0, "EXPAND": 0, "SUBMIT": 0}
        
        for step in range(20):
            # 固定策略：先 CALL 3次，再 EXPAND 3次，最后 SUBMIT
            if step < 3:
                action = np.array([1, 0, 0, 0, 0])  # CALL
            elif step < 6:
                action = np.array([2, 0, 0, 0, 0])  # EXPAND
            else:
                action = np.array([3, 0, 0, 0, 0])  # SUBMIT
            
            action_type = action[0]
            action_name = ["JUMP", "CALL", "EXPAND", "SUBMIT"][action_type]
            actions_taken[action_name] += 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            if terminated or truncated:
                break
        
        # 判断是否正确
        final_node = env.current_node
        final_node_name = getattr(evaluator.graph.nodes.get(final_node), "name", "") if final_node else ""
        ground_truth = evaluator._get_ground_truth(sample)
        is_correct = evaluator._is_correct(final_node_name, ground_truth)
        
        results.append({
            "correct": is_correct,
            "steps": step_count,
            "actions": actions_taken,
            "prediction": final_node_name,
        })
    
    return results


def evaluate_with_rl_strategy(evaluator, sample_indices, model_path):
    """使用 RL 策略"""
    model = PPO.load(model_path)
    
    results = []
    for idx in tqdm(sample_indices, desc="RL Strategy"):
        sample = evaluator.df.iloc[idx].to_dict()
        issue = sample.get("problem_statement", "")
        
        verifier = VerifierAgent(evaluator.graph, use_llm=evaluator.use_llm)
        reasoner = ReasonerAgent(evaluator.graph, evaluator.retriever, use_llm=evaluator.use_llm)
        
        env = FaultLocalizationEnv(
            graph=evaluator.graph,
            retriever=evaluator.retriever,
            bug_query=issue,
            max_steps=20,
            top_k_retrieval=5,
            reasoner=reasoner,
            verifier=verifier,
        )
        
        obs, _ = env.reset()
        step_count = 0
        actions_taken = {"JUMP": 0, "CALL": 0, "EXPAND": 0, "SUBMIT": 0}
        
        for step in range(20):
            action, _ = model.predict(obs, deterministic=True)
            action_type = action[0] if hasattr(action, '__getitem__') else action
            action_name = ["JUMP", "CALL", "EXPAND", "SUBMIT"][action_type]
            actions_taken[action_name] += 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            if terminated or truncated:
                break
        
        final_node = env.current_node
        final_node_name = getattr(evaluator.graph.nodes.get(final_node), "name", "") if final_node else ""
        ground_truth = evaluator._get_ground_truth(sample)
        is_correct = evaluator._is_correct(final_node_name, ground_truth)
        
        results.append({
            "correct": is_correct,
            "steps": step_count,
            "actions": actions_taken,
            "prediction": final_node_name,
        })
    
    return results


def main():
    print("=" * 60)
    print("策略对比实验")
    print("=" * 60)
    
    # 构建基础组件
    print("\n加载图和检索器...")
    builder = GraphBuilder()
    graph = builder.build_from_repo("data_storage/repos/astropy")
    
    corpus = {}
    for node_id, node in graph.nodes.items():
        name = getattr(node, "name", "") or ""
        doc = getattr(node, "doc", "") or ""
        file_path = getattr(node, "file_path", "") or ""
        code = getattr(node, "code", "") or ""
        corpus[node_id] = f"{name} {doc} {file_path} {str(code)[:300]}".strip()
    
    bm25 = SimpleBM25(corpus)
    retriever = Retriever(bm25, use_dense=True, dense_model_path="./dense_retriever_model")
    retriever.build_index(graph)
    
    # 创建评估器
    class SimpleEvaluator:
        def __init__(self, graph, retriever, df):
            self.graph = graph
            self.retriever = retriever
            self.df = df
            self.use_llm = True
        
        def _get_ground_truth(self, sample):
            import re
            ground_truth = []
            if "patch" in sample and sample["patch"]:
                patch = sample["patch"]
                file_matches = re.findall(r'diff --git a/(.+?) b/(.+?)\n', patch)
                for old_file, new_file in file_matches:
                    ground_truth.append(new_file)
                    file_name = new_file.split('/')[-1].replace('.py', '')
                    ground_truth.append(file_name)
                func_matches = re.findall(r'@@.*@@\s+def\s+(\w+)\s*\(', patch)
                for func in func_matches:
                    ground_truth.append(func)
            return list(set(ground_truth))
        
        def _is_correct(self, prediction, ground_truth):
            if not ground_truth or not prediction:
                return False
            prediction_lower = prediction.lower()
            for gt in ground_truth:
                if gt.lower() in prediction_lower or prediction_lower in gt.lower():
                    return True
            return False
    
    df = pd.read_parquet("data_storage/splits/test.parquet")
    evaluator = SimpleEvaluator(graph, retriever, df)
    
    sample_indices = list(range(20))  # 测试前 20 个样本
    
    # 测试固定策略
    print("\n1. 测试固定策略...")
    fixed_results = evaluate_with_fixed_strategy(evaluator, sample_indices)
    
    # 测试 RL 策略
    print("\n2. 测试 RL 策略...")
    rl_results = evaluate_with_rl_strategy(evaluator, sample_indices, "best_model_fast.zip")
    
    # 输出对比结果
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    
    fixed_accuracy = sum(r["correct"] for r in fixed_results) / len(fixed_results)
    rl_accuracy = sum(r["correct"] for r in rl_results) / len(rl_results)
    
    fixed_steps = sum(r["steps"] for r in fixed_results) / len(fixed_results)
    rl_steps = sum(r["steps"] for r in rl_results) / len(rl_results)
    
    fixed_calls = sum(r["actions"]["CALL"] for r in fixed_results) / len(fixed_results)
    rl_calls = sum(r["actions"]["CALL"] for r in rl_results) / len(rl_results)
    
    print(f"\n{'指标':<20} {'固定策略':<15} {'RL策略':<15}")
    print("-" * 50)
    print(f"{'准确率':<20} {fixed_accuracy*100:>14.1f}% {rl_accuracy*100:>14.1f}%")
    print(f"{'平均步数':<20} {fixed_steps:>14.1f} {rl_steps:>14.1f}")
    print(f"{'平均检索次数':<20} {fixed_calls:>14.1f} {rl_calls:>14.1f}")
    
    print("\n结论:")
    if rl_accuracy > fixed_accuracy:
        print(f"  ✅ RL 策略准确率提升 { (rl_accuracy - fixed_accuracy)*100:.1f}%")
    else:
        print(f"  ⚠️ RL 策略准确率未提升")
    
    if rl_calls < fixed_calls:
        print(f"  ✅ RL 策略减少检索次数 { (fixed_calls - rl_calls):.1f} 次")
    else:
        print(f"  ⚠️ RL 策略检索次数未减少")

if __name__ == "__main__":
    main()
