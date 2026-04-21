import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from graph.builder import GraphBuilder
from retrieval.retriever import Retriever
from agent.rl_env import FaultLocalizationEnv


class TrainingLoggerCallback(BaseCallback):
    """
    简单训练日志回调：
    - 定期打印 timestep
    - 打印最近 episode reward
    """

    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                last_ep = self.model.ep_info_buffer[-1]
                ep_reward = last_ep.get("r", None)
                ep_len = last_ep.get("l", None)
                print(
                    f"[Training] step={self.num_timesteps} | "
                    f"last_ep_reward={ep_reward} | last_ep_len={ep_len}"
                )
            else:
                print(f"[Training] step={self.num_timesteps}")
        return True


def main():
    # =========================
    # 1. 配置
    # =========================
    repo_path = "data_storage/repos/astropy"

    total_timesteps = 20000
    max_steps_per_episode = 10
    top_k_retrieval = 5
    query_mode = "weak"   # 可选: "oracle", "weak"
    seed = 42

    model_dir = "saved_models"
    log_dir = "logs"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # =========================
    # 2. 构图
    # =========================
    print("\n[1/5] Building graph...")
    builder = GraphBuilder()
    graph = builder.build_from_repo(repo_path)
    print(f"Graph built: {graph}")

    # =========================
    # 3. 建 retriever
    # =========================
    print("\n[2/5] Building retriever index...")
    retriever = Retriever(graph)
    retriever.build_index()
    print("Retriever index built.")

    # =========================
    # 4. 建环境
    # =========================
    print("\n[3/5] Creating RL environment...")
    env = FaultLocalizationEnv(
        graph=graph,
        retriever=retriever,
        bug_query=None,          # 让环境自动生成 query
        bug_node=None,           # 随机 bug node
        max_steps=max_steps_per_episode,
        top_k_retrieval=top_k_retrieval,
        seed=seed,
        query_mode=query_mode,
    )

    env = Monitor(env)
    print("Environment created.")

    # =========================
    # 5. 建模型
    # =========================
    print("\n[4/5] Initializing PPO...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=seed,
        tensorboard_log=log_dir,
        device="cpu",
    )
    print("PPO initialized.")

    # =========================
    # 6. 训练
    # =========================
    print("\n[5/5] Start training...")
    callback = TrainingLoggerCallback(check_freq=1000)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"ppo_fault_localizer_{query_mode}_{timestamp}")
    model.save(model_path)

    print(f"\nTraining finished.")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()