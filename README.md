# RL-NavFL: 基于强化学习的故障定位系统

> 融合 **强化学习导航策略** 与 **混合检索** 的智能故障定位系统，用于在大型代码仓库中自动定位软件缺陷。

---

## 📋 目录

- [系统简介](#系统简介)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [环境要求](#环境要求)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [训练模型](#训练模型)
- [评估与测试](#评估与测试)
- [项目结构](#项目结构)
- [常见问题](#常见问题)


---

## 系统简介

RL-NavFL 是一个智能故障定位系统，它将 **强化学习（RL）** 与 **代码图结构** 相结合，通过训练智能体在代码图中导航，自动定位导致软件缺陷的代码实体（函数、类或文件）。

**核心创新点：**

- RL 智能体学习 **何时检索、何时扩展、何时提交** 的导航策略。
- 混合检索（BM25 + Sentence-BERT）提供语义理解能力。
- 验证器提供反馈闭环，持续优化定位准确率。

---

## 核心特性

| 特性 | 说明 |
|---|---|
| 🔍 **混合检索** | BM25 关键词匹配 + Sentence-BERT 语义检索 |
| 🤖 **强化学习导航** | PPO 算法训练智能体在代码图中导航 |
| 📊 **SWE-bench 评估** | 支持在 SWE-bench Lite 基准上评估 |
| 🗂️ **完整代码图** | 全仓库代码图构建（19k+ 节点） |
| 🔄 **验证器反馈** | LLM 验证 + 奖励塑形 |
| 💻 **GPU/CPU 支持** | 支持 CUDA 加速 |

---

## 系统架构

```text
┌───────────────────────────────────────────────────────────────┐
│ 用户输入（Issue）                                             │
│ "Null pointer exception when accessing database"              │
└───────────────────────────────────────────────────────────────┘
                                ↓
┌───────────────────────────────────────────────────────────────┐
│ 阶段 1: LLM 认知（Reasoner）                                  │
│ 将自然语言转化为代码语义查询                                  │
└───────────────────────────────────────────────────────────────┘
                                ↓
┌───────────────────────────────────────────────────────────────┐
│ 阶段 2: 混合检索（Retriever）                                 │
│ BM25（稀疏）+ Sentence-BERT（稠密）                            │
│ 返回 Top-K 候选节点                                           │
└───────────────────────────────────────────────────────────────┘
                                ↓
┌───────────────────────────────────────────────────────────────┐
│ 阶段 3: RL 导航（Agent）                                      │
│                                                               │
│      ┌──────┐    ┌──────┐    ┌────────┐    ┌────────┐         │
│      │ JUMP │    │ CALL │    │ EXPAND │    │ SUBMIT │         │
│      └──────┘    └──────┘    └────────┘    └────────┘         │
│         ↓           ↓            ↓             ↓              │
│      跳转节点     语义检索      扩展邻居       提交结果        │
└───────────────────────────────────────────────────────────────┘
                                ↓
┌───────────────────────────────────────────────────────────────┐
│ 阶段 4: 验证器（Verifier）                                    │
│ LLM 辩论验证 + 最终判决                                       │
└───────────────────────────────────────────────────────────────┘
                                ↓
┌───────────────────────────────────────────────────────────────┐
│ 输出定位结果                                                  │
│ "astropy/config/tests/test_configs.py"                        │
│ "test_config_noastropy_fallback"                              │
└───────────────────────────────────────────────────────────────┘
```

---

## 环境要求

| 依赖 | 版本 | 用途 |
|---|---:|---|
| Python | 3.10+ | 运行环境 |
| PyTorch | 2.0+ | 深度学习框架 |
| stable-baselines3 | 2.0+ | PPO 算法实现 |
| sentence-transformers | 2.2+ | 语义检索模型 |
| Ollama | 0.1+ | LLM 服务（Reasoner/Verifier） |
| CUDA | 11.8+ | GPU 加速（可选） |

---

## 安装指南

### 1. 克隆仓库

```bash
git clone https://github.com/your-repo/fault_localization_rl.git
cd fault_localization_rl
```

### 2. 创建虚拟环境

```bash
conda create -n fault_rl python=3.10
conda activate fault_rl
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 安装 Ollama（LLM 服务）

```bash
# Linux / WSL
curl -fsSL https://ollama.ai/install.sh | sh

# 启动服务
ollama serve &

# 下载模型
ollama pull qwen2.5-coder:32b  # 推荐：代码专用模型

# 或
ollama pull llama3:8b           # 备选：通用模型
```

### 5. 下载语义检索模型

```bash
# 在有网络的机器上
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('./dense_retriever_model')
"

# 打包并传到服务器
tar -czf dense_retriever_model.tar.gz dense_retriever_model/
```

---

## 快速开始

### 1. 准备数据

```bash
# 划分训练/验证/测试集
python split_data_manual.py
```

### 2. 训练模型

```bash
# 快速训练（50k 步）
nohup python main.py --mode train \
  --repo_path "data_storage/repos/astropy" \
  --data_path "data_storage/splits/train.parquet" \
  --split train \
  --use_dense \
  --use_llm \
  --timesteps 50000 \
  --max_steps 20 \
  --ent_coef 0.2 \
  --learning_rate 0.001 \
  --model_output "best_model" \
  --device cuda \
  > training.log 2>&1 &
```

### 3. 推理定位

```bash
python main.py --mode inference \
  --repo_path "data_storage/repos/astropy" \
  --issue_text "Null pointer exception occurs when accessing the database." \
  --use_dense \
  --use_llm \
  --model_path "best_model.zip" \
  --max_steps 20 \
  --output_file "result.json"
```

### 4. 评估模型

```bash
python evaluate.py \
  --split test \
  --use_dense \
  --use_llm \
  --model_path "best_model.zip" \
  --experiment accuracy \
  --device cuda
```

---

## 配置说明

### 主要参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--mode` | `train` | 运行模式：`train` / `inference` |
| `--max_steps` | `20` | 每个 episode 最大步数 |
| `--timesteps` | `100000` | 总训练步数 |
| `--use_dense` | `False` | 启用语义检索 |
| `--use_llm` | `False` | 启用 LLM |
| `--ent_coef` | `0.01` | 熵系数（探索率） |
| `--learning_rate` | `0.0005` | 学习率 |
| `--device` | `cpu` | 计算设备：`cuda` / `cpu` |

### 混合检索权重

```python
# 在 main.py 中配置
retriever = Retriever(
    bm25,
    alpha=0.5,      # BM25 权重
    beta=0.5,       # Dense 权重
    use_dense=True,
)
```

### 奖励函数权重

```python
# 论文公式: r = α·r_progress + β·r_relevance + γ·r_efficiency + δ·r_verify + ε·r_final
self.reward_weights = {
    "progress": 0.30,     # 进度奖励
    "relevance": 0.20,    # 相关性奖励
    "efficiency": 0.15,   # 效率奖励
    "verify": 0.20,       # 验证器反馈
    "final": 0.15,        # 最终提交奖励
}
```

---

## 训练模型

### 完整训练命令

```bash
nohup python main.py --mode train \
  --repo_path "data_storage/repos/astropy" \
  --data_path "data_storage/splits/train.parquet" \
  --split train \
  --use_dense \
  --use_llm \
  --timesteps 300000 \
  --max_steps 30 \
  --ent_coef 0.1 \
  --learning_rate 0.0003 \
  --batch_size 128 \
  --n_steps 4096 \
  --model_output "best_model_300k" \
  --device cuda \
  --tensorboard_log "./logs" \
  > training.log 2>&1 &
```

### 监控训练

```bash
# 查看实时日志
tail -f training.log

# TensorBoard 可视化
tensorboard --logdir ./logs --port 6006
```

### 训练时间估算

| 步数 | 设备 | 预计时间 |
|---:|---|---:|
| 50k | GPU（fps=5） | ~3 小时 |
| 100k | GPU（fps=5） | ~6 小时 |
| 300k | GPU（fps=5） | ~17 小时 |

---

## 评估与测试

### 准确率评估

```bash
python evaluate.py \
  --split test \
  --use_dense \
  --use_llm \
  --model_path "best_model.zip" \
  --experiment accuracy \
  --device cuda
```

### 效率对比

```bash
python evaluate.py \
  --split test \
  --use_dense \
  --use_llm \
  --experiment efficiency \
  --device cuda
```

### 消融实验

```bash
python evaluate.py \
  --split test \
  --use_dense \
  --use_llm \
  --experiment ablation \
  --device cuda
```

### 快速测试（10 个样本）

```bash
python -c "
from evaluate import SWEBenchEvaluator

evaluator = SWEBenchEvaluator(
    data_path='data_storage/splits/test.parquet',
    repo_path='data_storage/repos/astropy',
    split='test',
    use_dense=True,
    use_llm=True,
    device='cuda',
    model_path='best_model.zip'
)

results = evaluator.evaluate_all(sample_indices=list(range(10)), use_rl=True)
print(f'Top-1 Acc: {results[\"is_correct\"].mean():.1%}')
"
```

---

## 项目结构

```text
fault_localization_rl/
├── agent/
│   ├── __init__.py
│   └── rl_env.py                  # RL 环境
├── retrieval/
│   ├── __init__.py
│   ├── bm25.py                    # BM25 检索
│   ├── retriever.py               # 混合检索器
│   ├── dense_retriever.py         # 语义检索器
│   └── vector_store.py            # TF-IDF 检索（备用）
├── reasoner/
│   └── reasoner_agent.py          # LLM 认知层
├── verifier/
│   └── debate_agent.py            # LLM 验证器
├── graph/
│   └── builder.py                 # 代码图构建
├── utils/
│   └── llm.py                     # LLM 统一接口
├── data_storage/
│   ├── repos/                     # 代码仓库
│   ├── splits/                    # 数据集划分
│   │   ├── train.parquet
│   │   ├── val.parquet
│   │   └── test.parquet
│   └── SWE-bench_Lite/            # SWE-bench 数据
├── dense_retriever_model/         # Sentence-BERT 模型
├── main.py                        # 主入口
├── evaluate.py                    # 评估脚本
├── split_data_manual.py           # 数据划分
└── requirements.txt               # 依赖列表
```

---

## 常见问题

### Q1: 模型只输出 CALL，从不 SUBMIT？

**原因：** CALL 奖励过高，模型选择“安全”动作。

**解决：** 降低 `_handle_call` 中的 `tool_reward`，提高 `_handle_submit` 中的命中奖励。

### Q2: Ollama 连接失败？

```bash
# 检查服务状态
ollama list

# 重启服务
ollama serve &

# 检查端口
curl http://localhost:11434/api/tags
```

### Q3: 语义检索模型下载失败？

```bash
# 使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载后传到服务器
```

