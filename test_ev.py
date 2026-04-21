from graph.builder import GraphBuilder
from retrieval.retriever import Retriever
from agent.rl_env import FaultLocalizationEnv

repo_path = "data_storage/repos/astropy"

builder = GraphBuilder()
graph = builder.build_from_repo(repo_path)

retriever = Retriever(graph)
retriever.build_index()

env = FaultLocalizationEnv(
    graph=graph,
    retriever=retriever,
    bug_query=None,  # 关键：不要再传固定 query
    query_mode="weak",  # 这里传入 weak 或 oracle 模式
    max_steps=10,
    top_k_retrieval=5,
    seed=42,
)

obs, info = env.reset()

print("=== RESET ===")
print("Obs:", obs)
print("Info:", info)

done = False
truncated = False

for i in range(5):
    action = i % 3
    obs, reward, done, truncated, info = env.step(action)

    print(f"\n=== STEP {i+1} ===")
    print("Action:", info["action_name"])
    print("Reward:", reward)
    print("Obs:", obs)
    print("Info:", info)

    if done or truncated:
        break