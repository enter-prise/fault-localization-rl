#!/usr/bin/env python3
"""验证修改后的系统是否能正常运行"""

import sys
import numpy as np
from graph.builder import GraphBuilder
from retrieval.bm25 import SimpleBM25
from retrieval.retriever import Retriever
from agent.rl_env import FaultLocalizationEnv
from reasoner.reasoner_agent import ReasonerAgent
from verifier.debate_agent import VerifierAgent

def main():
    print("=" * 60)
    print("System Verification")
    print("=" * 60)
    
    # 1. 构建图
    print("\n[1/6] Building graph...")
    try:
        builder = GraphBuilder()
        graph = builder.build_from_repo("data_storage/repos/astropy")
        print(f"  ✅ Graph built: {len(graph.nodes)} nodes")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False
    
    # 2. 构建检索器
    print("\n[2/6] Building retriever...")
    try:
        corpus = {nid: "test" for nid in graph.nodes}
        bm25 = SimpleBM25(corpus)
        retriever = Retriever(bm25)
        retriever.build_index(graph)
        print("  ✅ Retriever built")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False
    
    # 3. 创建 Reasoner 和 Verifier
    print("\n[3/6] Creating reasoner and verifier...")
    try:
        reasoner = ReasonerAgent(graph, retriever, use_llm=False)  # 不用 LLM 加快测试
        verifier = VerifierAgent(graph, use_llm=False)
        print("  ✅ Agents created")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False
    
    # 4. 创建环境
    print("\n[4/6] Creating environment...")
    try:
        env = FaultLocalizationEnv(
            graph=graph,
            retriever=retriever,
            bug_query="Null pointer exception when accessing the database.",
            max_steps=20,
            top_k_retrieval=5,
            reasoner=reasoner,
            verifier=verifier,
            seed=42,
        )
        print(f"  ✅ Environment created")
        print(f"     Observation space: {env.observation_space}")
        print(f"     Action space: {env.action_space}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False
    
    # 5. 测试各种动作
    print("\n[5/6] Testing actions...")
    obs, info = env.reset()
    
    actions_to_test = [
        (np.array([0, 0, 0, 0, 0]), "JUMP"),
        (np.array([1, 0, 0, 0, 0]), "CALL"),
        (np.array([2, 0, 0, 0, 0]), "EXPAND"),
        (np.array([3, 0, 0, 0, 0]), "SUBMIT"),
    ]
    
    for action, name in actions_to_test:
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  ✅ {name}: reward={reward:.3f}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            return False
    
    # 6. 测试完整 episode
    print("\n[6/6] Testing full episode (10 steps)...")
    obs, info = env.reset()
    action_counts = {"JUMP": 0, "CALL": 0, "EXPAND": 0, "SUBMIT": 0}
    
    for step in range(10):
        # 轮流尝试不同动作
        action_type = step % 4
        action = np.array([action_type, 0, 0, 0, 0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        action_name = ["JUMP", "CALL", "EXPAND", "SUBMIT"][action_type]
        action_counts[action_name] += 1
        
        print(f"  Step {step+1}: {action_name} -> reward={reward:.3f}")
        
        if terminated or truncated:
            break
    
    print("\n" + "=" * 60)
    print("VERIFICATION RESULT")
    print("=" * 60)
    print(f"Action distribution: {action_counts}")
    print("\n✅ All tests passed! System is ready for training.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)