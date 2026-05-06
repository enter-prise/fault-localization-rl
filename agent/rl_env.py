import math
import random
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FaultLocalizationEnv(gym.Env):
    """
    优化版 RL 环境 - 让每步都有明确的学习信号
    """

    metadata = {"render_modes": ["human"]}

    ACTION_JUMP = 0
    ACTION_CALL = 1
    ACTION_EXPAND = 2
    ACTION_SUBMIT = 3

    TOOL_SEMANTIC_SCOUT = 0
    TOOL_CODE_EXPLORER = 1
    TOOL_CONTEXT_PROBE = 2

    ACTION_NAMES = {
        ACTION_JUMP: "jump_to",
        ACTION_CALL: "call_tool",
        ACTION_EXPAND: "expand_subgraph",
        ACTION_SUBMIT: "submit_result",
    }

    TOOL_NAMES = {
        TOOL_SEMANTIC_SCOUT: "semantic_scout",
        TOOL_CODE_EXPLORER: "code_explorer",
        TOOL_CONTEXT_PROBE: "context_probe",
    }

    VERDICT_ACCEPT = "accept"
    VERDICT_REJECT = "reject"
    VERDICT_UNCERTAIN = "uncertain"
    VERDICT_UNKNOWN = "unknown"

    def __init__(
        self,
        graph,
        retriever,
        bug_query: Optional[str] = None,
        bug_node: Optional[str] = None,
        max_steps: int = 30,  # 增加步数上限到30
        top_k_retrieval: int = 5,
        max_jump_candidates: int = 8,
        max_expand_hop: int = 3,
        max_candidate_pool: int = 20,
        seed: Optional[int] = None,
        query_mode: str = "weak",
        reasoner=None,
        verifier=None,
        auto_terminate_on_exact_hit: bool = False,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        self.graph = graph
        self.retriever = retriever
        self.reasoner = reasoner
        self.verifier = verifier
        self.use_guided_expansion = True
        self.query_mode = query_mode
        self.user_bug_query = bug_query
        self.fixed_bug_node = bug_node

        self.max_steps = max_steps
        self.top_k_retrieval = max(1, int(top_k_retrieval))
        self.max_jump_candidates = max(1, int(max_jump_candidates))
        self.max_expand_hop = max(1, int(max_expand_hop))
        self.max_candidate_pool = max(self.top_k_retrieval, int(max_candidate_pool))
        self.auto_terminate_on_exact_hit = auto_terminate_on_exact_hit

        # 优化后的奖励权重
        self.reward_weights = {
            "progress": 0.40,      # 提高
            "relevance": 0.35,     # 提高
            "efficiency": 0.05,    # 降低
            "verify": 0.10,
            "final": 0.10,
        }

        # 优化后的奖励阈值
        self.reward_thresholds = {
            "final_hit_top1": 2.0,
            "final_hit_top3": 1.0,
            "final_hit_top5": 0.5,
            "final_miss": -1.0,
            "revisit_penalty": -0.05,  # 降低重复访问惩罚
            "step_penalty": -0.002,    # 大幅降低步数惩罚（从-0.02到-0.002）
            "explore_bonus": 0.12,     # 探索新节点奖励
            "max_reward": 1.5,
            "min_reward": -1.0,
        }

        # 验证器反馈奖励参数
        self.verifier_reward_params = {
            "accept_base": 0.5,
            "accept_scale": 0.5,
            "reject_base": -0.3,
            "reject_scale": -0.5,
            "uncertain_scale": -0.2,
        }

        self.rng = random.Random(seed)

        self.node_ids: List[str] = list(self.graph.nodes.keys())
        if not self.node_ids:
            raise ValueError("Graph is empty. Cannot create environment.")

        # episode state
        self.current_node: Optional[str] = None
        self.bug_node: Optional[str] = None
        self.bug_query: Optional[str] = None
        self.steps = 0

        self.history: List[str] = []
        self.visited_set = set()
        self.trajectory: List[Dict[str, Any]] = []

        self.last_retrieval_results: List[Dict[str, Any]] = []
        self.last_reasoner_choice: Optional[Tuple[str, float]] = None
        self.last_reasoner_trace: Optional[Dict[str, Any]] = None
        self.last_verifier_result: Optional[Dict[str, Any]] = None
        self.last_tool_result: Optional[Dict[str, Any]] = None
        self.last_expand_result: Optional[Dict[str, Any]] = None

        self.candidate_pool: List[Dict[str, Any]] = []
        self.last_submit_candidates: List[Dict[str, Any]] = []

        self.action_counter = Counter()
        self.tool_counter = Counter()
        
        self.last_reward_breakdown: Dict[str, float] = {}

        # MultiDiscrete 动作空间
        self.action_space = spaces.MultiDiscrete([
            4,
            self.max_jump_candidates,
            3,
            self.max_expand_hop,
            self.top_k_retrieval,
        ])

        # 32 维状态向量
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(32,),
            dtype=np.float32,
        )

    # ==================================================================
    # 辅助函数
    # ==================================================================

    def _tokenize_simple(self, text: str) -> List[str]:
        if not text:
            return []
        import re
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9]+', ' ', text)
        return text.split()

    def _get_node_relevance_score(self, node_id: str) -> float:
        """获取指定节点的相关性分数"""
        if node_id is None:
            return 0.0
        for item in self.last_retrieval_results:
            if item.get("entity_id") == node_id:
                return min(max(float(item.get("relevance_score", 0.0)), 0.0), 1.0)
        for item in self.candidate_pool:
            if item.get("entity_id") == node_id:
                return min(max(float(item.get("relevance_score", 0.0)), 0.0), 1.0)
        return 0.0

    def _compute_graph_distance(self, node_id: str, target_id: str) -> float:
        if node_id == target_id:
            return 1.0
        if node_id is None or target_id is None:
            return 0.0
        
        visited = set()
        queue = deque([(node_id, 0)])
        
        while queue:
            curr_id, dist = queue.popleft()
            if curr_id == target_id:
                return max(0.0, 1.0 - dist / 10.0)
            if curr_id in visited:
                continue
            visited.add(curr_id)
            for neighbor in self._safe_get_neighbors(curr_id):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        return 0.0

    def _compute_semantic_similarity(self, query: str, node_id: str) -> float:
        if not query or node_id is None:
            return 0.0
        
        node = self.graph.nodes.get(node_id)
        if node is None:
            return 0.0
        
        node_text = " ".join([
            str(getattr(node, "name", "")),
            str(getattr(node, "type", "")),
            str(getattr(node, "doc", ""))[:200],
            str(getattr(node, "file_path", "")),
        ]).lower()
        
        query_tokens = set(self._tokenize_simple(query))
        node_tokens = set(self._tokenize_simple(node_text))
        
        if not query_tokens:
            return 0.0
        
        overlap = len(query_tokens & node_tokens)
        union = len(query_tokens | node_tokens)
        jaccard = overlap / union if union > 0 else 0.0
        
        keywords = ["null", "pointer", "exception", "error", "database", "access", "connection", "query"]
        keyword_bonus = 0.0
        query_lower = query.lower()
        for kw in keywords:
            if kw in query_lower and kw in node_text:
                keyword_bonus += 0.03
        
        return min(1.0, jaccard + keyword_bonus)

    def _compute_progress_reward(self, before_node: str, after_node: str) -> float:
        if self.bug_node is None:
            return 0.0
        old_distance = self._compute_graph_distance(before_node, self.bug_node)
        new_distance = self._compute_graph_distance(after_node, self.bug_node)
        return new_distance - old_distance

    def _compute_relevance_improvement(self, before_node: str, after_node: str) -> float:
        """计算相关性改进奖励"""
        old_relevance = self._get_node_relevance_score(before_node)
        new_relevance = self._get_node_relevance_score(after_node)
        improvement = max(0, new_relevance - old_relevance)
        return improvement * 0.8  # 相关性进步奖励系数

    def _compute_node_type_bonus(self, node_id: str) -> float:
        """根据节点类型给奖励"""
        if node_id is None:
            return 0.0
        node = self.graph.nodes.get(node_id)
        if node is None:
            return 0.0
        node_type = getattr(node, "type", "")
        if node_type == "function":
            return 0.10
        elif node_type == "method":
            return 0.08
        elif node_type == "class":
            return 0.05
        elif node_type == "file":
            return 0.02
        return 0.0

    def _compute_exploration_bonus(self, node_id: str) -> float:
        """计算探索奖励"""
        if node_id not in self.visited_set:
            return 0.15  # 首次访问高奖励
        elif self.history.count(node_id) == 2:
            return 0.03  # 第二次访问少量奖励
        else:
            return -0.03  # 重复访问轻微惩罚

    def _compute_verifier_reward(self, verifier_result: Dict[str, Any]) -> float:
        if verifier_result is None:
            return 0.0
        verdict = verifier_result.get("verdict", self.VERDICT_UNKNOWN)
        confidence = verifier_result.get("confidence", 0.0)
        confidence = min(max(confidence, 0.0), 1.0)
        
        params = self.verifier_reward_params
        if verdict == self.VERDICT_ACCEPT:
            return params["accept_base"] + params["accept_scale"] * confidence
        elif verdict == self.VERDICT_REJECT:
            return params["reject_base"] + params["reject_scale"] * confidence
        elif verdict == self.VERDICT_UNCERTAIN:
            return params["uncertain_scale"] * confidence
        return 0.0

    # ==================================================================
    # Reset
    # ==================================================================

    def _sample_bug_case(self):
        if self.fixed_bug_node is not None:
            if self.fixed_bug_node not in self.graph.nodes:
                raise ValueError(f"bug_node {self.fixed_bug_node} not found in graph.")
            self.bug_node = self.fixed_bug_node
        else:
            candidate_bug_nodes = []
            for node_id, node in self.graph.nodes.items():
                node_type = getattr(node, "type", None)
                if node_type in {"function", "class", "file", "method"}:
                    candidate_bug_nodes.append(node_id)
            if not candidate_bug_nodes:
                raise ValueError("No valid candidate bug nodes found in graph.")
            self.bug_node = self.rng.choice(candidate_bug_nodes)
        self.bug_query = self._resolve_query()

    def _resolve_query(self) -> str:
        if self.query_mode == "manual" and self.user_bug_query:
            return self.user_bug_query
        if self.user_bug_query and self.query_mode != "manual":
            return self.user_bug_query
        return self._generate_query_from_bug_node(self.bug_node)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)

        self.steps = 0
        self.history = []
        self.visited_set = set()
        self.trajectory = []

        self.last_retrieval_results = []
        self.last_reasoner_choice = None
        self.last_reasoner_trace = None
        self.last_verifier_result = None
        self.last_tool_result = None
        self.last_expand_result = None
        self.last_submit_candidates = []

        self.candidate_pool = []
        self.action_counter = Counter()
        self.tool_counter = Counter()
        self.last_reward_breakdown = {}

        self._sample_bug_case()
        self.current_node = self.rng.choice(self.node_ids)
        self._visit(self.current_node)
        self._bootstrap_candidate_pool()

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    # ==================================================================
    # Step - 核心改进在这里
    # ==================================================================

    def step(self, action):
        if self.current_node is None:
            raise ValueError("Environment must be reset before stepping.")

        parsed = self._parse_action(action)
        action_type = parsed["action_type"]
        jump_idx = parsed["jump_idx"]
        tool_idx = parsed["tool_idx"]
        expand_hop = parsed["expand_hop"]
        submit_topk = parsed["submit_topk"]

        self.steps += 1
        action_name = self._action_name(action_type)
        self.action_counter[action_name] += 1

        terminated = False
        truncated = False
        action_detail = {}
        before_node = self.current_node
        delta_reward = 0.0

        # 执行动作
        if action_type == self.ACTION_JUMP:
            delta_reward, action_detail = self._handle_jump(jump_idx)
        elif action_type == self.ACTION_CALL:
            delta_reward, action_detail = self._handle_call(tool_idx)
        elif action_type == self.ACTION_EXPAND:
            delta_reward, action_detail = self._handle_expand(expand_hop)
        elif action_type == self.ACTION_SUBMIT:
            delta_reward, terminated, action_detail = self._handle_submit(submit_topk)
        else:
            delta_reward = -1.0
            action_detail = {"error": "invalid_action_type"}

        if self.current_node is not None:
            self._visit(self.current_node)

        # ============================================================
        # 改进的奖励计算 - 让每步都有意义
        # ============================================================
        
        if action_type == self.ACTION_SUBMIT:
            reward = delta_reward
            self.last_reward_breakdown = action_detail.get("reward_breakdown", {})
        else:
            # 1. 基础步数成本（已大幅降低）
            step_cost = self.reward_thresholds["step_penalty"]  # -0.002
            
            # 2. 进度奖励（距离 bug 节点的变化）
            r_progress = self._compute_progress_reward(before_node, self.current_node)
            
            # 3. 语义进步奖励（新增 - 关键改进）
            r_relevance_improvement = self._compute_relevance_improvement(before_node, self.current_node)
            
            # 4. 探索奖励
            r_exploration = self._compute_exploration_bonus(self.current_node)
            
            # 5. 节点类型奖励
            r_type = self._compute_node_type_bonus(self.current_node)
            
            # 6. 动作本身奖励
            r_action = delta_reward
            
            # 组合奖励
            reward = (
                step_cost +
                self.reward_weights["progress"] * r_progress +
                r_relevance_improvement +      # 直接加，权重高
                r_exploration +
                r_type +
                r_action
            )
            
            # 添加基于检索排名的奖励（如果排名高）
            rank_score = self._get_retrieval_rank_bonus(self.current_node)
            reward += rank_score
            
            # 裁剪
            reward = max(self.reward_thresholds["min_reward"], 
                        min(self.reward_thresholds["max_reward"], reward))
            
            self.last_reward_breakdown = {
                "step_cost": round(step_cost, 4),
                "r_progress": round(r_progress, 4),
                "r_relevance_improvement": round(r_relevance_improvement, 4),
                "r_exploration": round(r_exploration, 4),
                "r_type": round(r_type, 4),
                "r_action": round(r_action, 4),
                "rank_bonus": round(rank_score, 4),
                "total": round(reward, 4),
            }

        # 到达真实 bug 节点奖励
        if self.current_node == self.bug_node and not terminated:
            reward += 0.8
            if self.auto_terminate_on_exact_hit:
                terminated = True
                reward += 1.0

        if self.steps >= self.max_steps and not terminated:
            truncated = True

        transition = {
            "step": self.steps,
            "from_node": before_node,
            "to_node": self.current_node,
            "action_type": action_type,
            "action_name": action_name,
            "jump_idx": jump_idx,
            "tool_idx": tool_idx,
            "expand_hop": expand_hop if action_type == self.ACTION_EXPAND else None,
            "submit_topk": submit_topk if action_type == self.ACTION_SUBMIT else None,
            "reward": reward,
            "reward_breakdown": self.last_reward_breakdown,
            "detail": action_detail,
        }
        self.trajectory.append(transition)

        obs = self._get_observation()
        info = self._get_info()
        info["transition"] = transition
        info["reward"] = reward
        info["reward_breakdown"] = self.last_reward_breakdown
        info["terminated"] = terminated
        info["truncated"] = truncated
        
        return obs, reward, terminated, truncated, info

    def _get_retrieval_rank_bonus(self, node_id: str) -> float:
        """根据检索排名给奖励"""
        if not self.last_retrieval_results:
            return 0.0
        for rank, result in enumerate(self.last_retrieval_results[:5]):
            if result.get("entity_id") == node_id:
                if rank == 0:
                    return 0.3
                elif rank == 1:
                    return 0.2
                elif rank == 2:
                    return 0.1
        return 0.0

    # ==================================================================
    # Action Handlers
    # ==================================================================

    def _handle_jump(self, jump_idx: int) -> Tuple[float, Dict[str, Any]]:
        jump_candidates = self._get_jump_candidates()
        if not jump_candidates:
            return -0.2, {"status": "no_jump_candidates"}

        selected = jump_candidates[min(jump_idx, len(jump_candidates) - 1)]
        target_node = selected["entity_id"]
        if target_node not in self.graph.nodes:
            return -0.2, {"status": "invalid_target", "selected": selected}

        self.current_node = target_node
        
        jump_reward = 0.03  # 基础跳转奖励
        
        if target_node not in self.visited_set:
            jump_reward += 0.08
        
        relevance_score = selected.get("relevance_score", 0.0)
        if relevance_score > 0.5:
            jump_reward += 0.08 * relevance_score

        return jump_reward, {"status": "ok", "selected": selected, "to_node": self.current_node}

    def _handle_call(self, tool_idx: int) -> Tuple[float, Dict[str, Any]]:
        tool_idx = int(tool_idx)
        tool_name = self._tool_name(tool_idx)
        self.tool_counter[tool_name] += 1

        if tool_idx == self.TOOL_SEMANTIC_SCOUT:
            results = self._tool_semantic_scout(self.bug_query, self.top_k_retrieval)
        elif tool_idx == self.TOOL_CODE_EXPLORER:
            results = self._tool_code_explorer(self.current_node, self.top_k_retrieval)
        elif tool_idx == self.TOOL_CONTEXT_PROBE:
            results = self._tool_context_probe(self.current_node, self.top_k_retrieval)
        else:
            return -0.3, {"status": "unknown_tool", "tool_idx": tool_idx}

        self.last_tool_result = {"tool_idx": tool_idx, "tool_name": tool_name, "results": results}

        if not results:
            return -0.2, {"status": "empty_results", "tool_name": tool_name}

        self.last_retrieval_results = results
        self._merge_into_candidate_pool(results)

        chosen = self._choose_candidate_after_tool(results)
        if chosen is None:
            return -0.1, {"status": "no_chosen_candidate", "tool_name": tool_name}

        self.current_node = chosen["entity_id"]

        tool_reward = 0.12  # 基础工具调用奖励
        
        if chosen and chosen["entity_id"] not in self.visited_set:
            tool_reward += 0.3
        
        if len(results) >= 3:
            tool_reward += 0.04
        
        relevance_score = chosen.get("relevance_score", 0.0)
        if relevance_score > 0.5:
            tool_reward += 0.08 * relevance_score

        return tool_reward, {"status": "ok", "tool_name": tool_name, "chosen": chosen, "to_node": self.current_node}
    def _handle_expand(self, hop_k: int) -> Tuple[float, Dict[str, Any]]:
        if self.current_node is None:
            return -0.2, {"status": "no_current_node"}

        nodes_in_subgraph = self._get_k_hop_nodes(self.current_node, hop_k)
        if not nodes_in_subgraph:
            return -0.1, {"status": "empty_subgraph", "hop_k": hop_k}

        # ============================================================
        # 引导式扩展：根据 query 相关性筛选邻居（避免扩展太多无关节点）
        # ============================================================
        if self.bug_query and len(nodes_in_subgraph) > 15:
            scored_neighbors = []
            for nid in nodes_in_subgraph:
                node = self.graph.nodes.get(nid)
                if node is None:
                    continue
                node_text = " ".join([
                    str(getattr(node, "name", "")),
                    str(getattr(node, "type", "")),
                    str(getattr(node, "doc", ""))[:100],
                    str(getattr(node, "file_path", "")),
                ])
                score = self._compute_semantic_similarity(self.bug_query, node_text)
                scored_neighbors.append((nid, score))
            
            scored_neighbors.sort(key=lambda x: x[1], reverse=True)
            nodes_in_subgraph = [nid for nid, _ in scored_neighbors[:15]]

        expanded_candidates = []
        for nid in nodes_in_subgraph:
            if nid not in self.graph.nodes:
                continue
            expanded_candidates.append(self._candidate_from_node(nid, 0.25, "expand"))

        self._merge_into_candidate_pool(expanded_candidates)

        new_candidates_count = len(expanded_candidates)
        new_nodes_discovered = sum(1 for c in expanded_candidates if c["entity_id"] not in self.visited_set)

        self.last_expand_result = {
            "center_node": self.current_node,
            "hop_k": hop_k,
            "subgraph_size": len(nodes_in_subgraph),
            "candidates_added": new_candidates_count,
            "new_nodes_discovered": new_nodes_discovered,
        }

        # 扩展奖励
        expand_reward = 0.06
        
        if new_candidates_count > 0:
            expand_reward += min(0.15, new_candidates_count * 0.04)
        
        if new_nodes_discovered > 0:
            expand_reward += min(0.20, new_nodes_discovered * 0.06)
        
        expand_reward += 0.03 * hop_k

        # 尝试跳转到最佳未访问候选
        best_fresh = self._pick_best_fresh_candidate(self.candidate_pool)
        jumped = None
        if best_fresh is not None and best_fresh["entity_id"] != self.current_node:
            jumped = best_fresh
            self.current_node = best_fresh["entity_id"]
            expand_reward += 0.12

        detail = {
            "status": "ok",
            "hop_k": hop_k,
            "subgraph_size": len(nodes_in_subgraph),
            "candidates_added": new_candidates_count,
            "new_nodes_discovered": new_nodes_discovered,
            "jumped_to": jumped,
            "to_node": self.current_node,
            "expand_reward": expand_reward,
        }
        
        return expand_reward, detail

    def _handle_submit(self, submit_topk: int) -> Tuple[float, bool, Dict[str, Any]]:
        candidates = self._build_submission_candidates(submit_topk)
        self.last_submit_candidates = candidates

        if not candidates:
            self.last_verifier_result = {"verdict": self.VERDICT_REJECT, "confidence": 0.0}
            return -2.0, True, {"status": "empty_submission"}

        submitted_ids = [c["entity_id"] for c in candidates]
        hit_top1 = submitted_ids[0] == self.bug_node
        hit_topk = self.bug_node in submitted_ids

        verifier_result = self._call_verifier(candidates)
        self.last_verifier_result = verifier_result
        r_verify = self._compute_verifier_reward(verifier_result)
        
        if hit_top1:
            r_final = self.reward_thresholds["final_hit_top1"]
        elif hit_topk:
            r_final = self.reward_thresholds["final_hit_top3"]
        else:
            r_final = self.reward_thresholds["final_miss"]

        total_reward = r_verify + r_final

        return total_reward, True, {
            "status": "ok",
            "submitted_ids": submitted_ids,
            "hit_top1": hit_top1,
            "hit_topk": hit_topk,
            "reward_breakdown": {"r_verify": round(r_verify, 3), "r_final": round(r_final, 3)},
        }

    # ==================================================================
    # 以下方法保持不变（Observation, Info, Graph helpers 等）
    # ==================================================================

    def _get_observation(self) -> np.ndarray:
        if self.current_node is None:
            return np.zeros((32,), dtype=np.float32)

        node = self.graph.nodes[self.current_node]
        node_type = getattr(node, "type", "")
        node_name = str(getattr(node, "name", "") or "")
        node_path = str(getattr(node, "file_path", "") or "")
        node_doc = str(getattr(node, "doc", "") or "")

        type_repo = 1.0 if node_type in {"repository", "repo"} else 0.0
        type_dir = 1.0 if node_type in {"directory", "dir"} else 0.0
        type_file = 1.0 if node_type == "file" else 0.0
        type_class = 1.0 if node_type == "class" else 0.0
        type_function = 1.0 if node_type in {"function", "method"} else 0.0
        known_type = max(type_repo, type_dir, type_file, type_class, type_function)
        type_other = 1.0 - known_type

        name_len_norm = min(len(node_name) / 64.0, 1.0)
        path_len_norm = min(len(node_path) / 128.0, 1.0)
        doc_len_norm = min(len(node_doc.split()) / 80.0, 1.0)
        current_relevance_norm = min(self._get_current_retrieval_score(), 1.0)
        revisit_flag = 1.0 if self.history.count(self.current_node) > 1 else 0.0

        total_neighbors = len(self._safe_get_neighbors(self.current_node))
        contains_neighbors = len(self._safe_get_neighbors(self.current_node, edge_type="contains"))
        calls_neighbors = len(self._safe_get_neighbors(self.current_node, edge_type="calls"))
        imports_neighbors = len(self._safe_get_neighbors(self.current_node, edge_type="imports"))
        other_neighbors = max(total_neighbors - contains_neighbors - calls_neighbors - imports_neighbors, 0)

        total_neighbors_norm = min(total_neighbors / 20.0, 1.0)
        contains_neighbors_norm = min(contains_neighbors / 10.0, 1.0)
        calls_neighbors_norm = min(calls_neighbors / 10.0, 1.0)
        imports_neighbors_norm = min(imports_neighbors / 10.0, 1.0)
        other_neighbors_norm = min(other_neighbors / 10.0, 1.0)

        k2_nodes = self._get_k_hop_nodes(self.current_node, 2)
        k2_size_norm = min(len(k2_nodes) / 30.0, 1.0)
        k2_unvisited_ratio = 0.0
        if k2_nodes:
            k2_unvisited_ratio = min(sum(1 for nid in k2_nodes if nid not in self.visited_set) / max(len(k2_nodes), 1), 1.0)
        candidate_pool_size_norm = min(len(self.candidate_pool) / max(self.max_candidate_pool, 1), 1.0)
        best_pool_score_norm = min(self._best_candidate_score(self.candidate_pool), 1.0)

        jump_ratio = self.action_counter[self._action_name(self.ACTION_JUMP)] / max(self.steps, 1)
        call_ratio = self.action_counter[self._action_name(self.ACTION_CALL)] / max(self.steps, 1)
        expand_ratio = self.action_counter[self._action_name(self.ACTION_EXPAND)] / max(self.steps, 1)
        submit_ratio = self.action_counter[self._action_name(self.ACTION_SUBMIT)] / max(self.steps, 1)

        verdict_accept = 0.0
        verdict_reject = 0.0
        verdict_uncertain = 0.0
        verdict_confidence = 0.0
        if self.last_verifier_result is not None:
            verdict = self.last_verifier_result.get("verdict", self.VERDICT_UNKNOWN)
            verdict_confidence = min(max(float(self.last_verifier_result.get("confidence", 0.0)), 0.0), 1.0)
            if verdict == self.VERDICT_ACCEPT:
                verdict_accept = 1.0
            elif verdict == self.VERDICT_REJECT:
                verdict_reject = 1.0
            elif verdict == self.VERDICT_UNCERTAIN:
                verdict_uncertain = 1.0

        last_tool_semantic = 0.0
        last_tool_explorer = 0.0
        last_tool_context = 0.0
        if self.last_tool_result is not None:
            tool_name = self.last_tool_result.get("tool_name", "")
            if tool_name == self._tool_name(self.TOOL_SEMANTIC_SCOUT):
                last_tool_semantic = 1.0
            elif tool_name == self._tool_name(self.TOOL_CODE_EXPLORER):
                last_tool_explorer = 1.0
            elif tool_name == self._tool_name(self.TOOL_CONTEXT_PROBE):
                last_tool_context = 1.0

        query_len_norm = min(len((self.bug_query or "").split()) / 40.0, 1.0)
        step_ratio = self.steps / max(self.max_steps, 1)
        visited_ratio = min(len(self.visited_set) / max(len(self.node_ids), 1), 1.0)
        current_in_pool = 1.0 if any(c["entity_id"] == self.current_node for c in self.candidate_pool) else 0.0

        observation = np.array([
            type_repo, type_dir, type_file, type_class, type_function, type_other,
            name_len_norm, path_len_norm, doc_len_norm, current_relevance_norm, revisit_flag,
            total_neighbors_norm, contains_neighbors_norm, calls_neighbors_norm,
            imports_neighbors_norm, other_neighbors_norm,
            k2_size_norm, k2_unvisited_ratio, candidate_pool_size_norm, best_pool_score_norm,
            jump_ratio, call_ratio, expand_ratio, submit_ratio,
            verdict_accept, verdict_reject, verdict_uncertain, verdict_confidence,
            query_len_norm, step_ratio, visited_ratio, current_in_pool,
        ], dtype=np.float32)

        return observation[:32]

    def _get_current_retrieval_score(self) -> float:
        if self.last_reasoner_choice is not None and self.last_reasoner_choice[0] == self.current_node:
            return min(max(float(self.last_reasoner_choice[1]), 0.0), 1.0)
        for item in self.last_retrieval_results:
            if item.get("entity_id") == self.current_node:
                return min(max(float(item.get("relevance_score", 0.0)), 0.0), 1.0)
        for item in self.candidate_pool:
            if item.get("entity_id") == self.current_node:
                return min(max(float(item.get("relevance_score", 0.0)), 0.0), 1.0)
        return 0.0

    def _parse_action(self, action) -> Dict[str, int]:
        if isinstance(action, (int, np.integer)):
            action = int(action)
            if action == 0:
                return {"action_type": self.ACTION_JUMP, "jump_idx": 0, "tool_idx": self.TOOL_SEMANTIC_SCOUT, "expand_hop": 1, "submit_topk": 1}
            if action == 1:
                return {"action_type": self.ACTION_CALL, "jump_idx": 0, "tool_idx": self.TOOL_SEMANTIC_SCOUT, "expand_hop": 1, "submit_topk": 1}
            if action == 2:
                return {"action_type": self.ACTION_EXPAND, "jump_idx": 0, "tool_idx": self.TOOL_SEMANTIC_SCOUT, "expand_hop": 1, "submit_topk": 1}
            if action == 3:
                return {"action_type": self.ACTION_SUBMIT, "jump_idx": 0, "tool_idx": self.TOOL_SEMANTIC_SCOUT, "expand_hop": 1, "submit_topk": 1}

        if isinstance(action, np.ndarray):
            action = action.tolist()
        if not isinstance(action, (list, tuple)) or len(action) < 5:
            raise ValueError("Action must be int or list/tuple of length 5")

        action_type = int(action[0])
        jump_idx = int(action[1])
        tool_idx = int(action[2])
        expand_hop = int(action[3]) + 1
        submit_topk = int(action[4]) + 1

        return {
            "action_type": action_type,
            "jump_idx": jump_idx,
            "tool_idx": tool_idx,
            "expand_hop": max(1, min(expand_hop, self.max_expand_hop)),
            "submit_topk": max(1, min(submit_topk, self.top_k_retrieval)),
        }

    def _bootstrap_candidate_pool(self):
        if not self.bug_query:
            return
        initial = self._tool_semantic_scout(self.bug_query, self.top_k_retrieval)
        self.last_retrieval_results = initial
        self._merge_into_candidate_pool(initial)

    def _tool_semantic_scout(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        raw_results = self._call_retriever(query=query, top_k=top_k)
        return self._canonicalize_results(raw_results, default_source="semantic")

    def _tool_code_explorer(self, node_id: Optional[str], top_k: int) -> List[Dict[str, Any]]:
        if node_id is None or node_id not in self.graph.nodes:
            return []
        neighbors = self._safe_get_neighbors(node_id)
        candidates = []
        for nid in neighbors[: max(top_k * 2, 6)]:
            candidates.append(self._candidate_from_node(nid, 0.35, "graph_explorer"))
        self.last_reasoner_choice = None
        self.last_reasoner_trace = None
        if self.reasoner is not None and candidates:
            reranked = self._try_reasoner_rerank(self.bug_query or "", candidates)
            if reranked:
                candidates = reranked
        return candidates[:top_k]

    def _tool_context_probe(self, node_id: Optional[str], top_k: int) -> List[Dict[str, Any]]:
        if node_id is None or node_id not in self.graph.nodes:
            return []
        node = self.graph.nodes[node_id]
        local_query_parts = [
            getattr(node, "name", ""),
            getattr(node, "type", ""),
            getattr(node, "file_path", ""),
            getattr(node, "doc", ""),
        ]
        local_query = " ".join([str(x) for x in local_query_parts if x]).strip()
        if not local_query:
            local_query = self.bug_query or ""
        raw_results = self._call_retriever(query=local_query, top_k=top_k)
        return self._canonicalize_results(raw_results, default_source="context")

    def _call_retriever(self, query: str, top_k: int):
        if hasattr(self.retriever, "hierarchical_retrieve"):
            try:
                return self.retriever.hierarchical_retrieve(query, top_k=top_k)
            except TypeError:
                try:
                    return self.retriever.hierarchical_retrieve(query)
                except Exception:
                    pass
            except Exception:
                pass
        if hasattr(self.retriever, "retrieve_with_score"):
            try:
                return self.retriever.retrieve_with_score(query, top_k=top_k)
            except TypeError:
                try:
                    return self.retriever.retrieve_with_score(query, self.graph)[:top_k]
                except Exception:
                    pass
            except Exception:
                pass
        if hasattr(self.retriever, "retrieve"):
            try:
                return self.retriever.retrieve(query, top_k=top_k)
            except TypeError:
                try:
                    return self.retriever.retrieve(query)[:top_k]
                except Exception:
                    pass
            except Exception:
                pass
        return []

    def _canonicalize_results(self, raw_results, default_source: str) -> List[Dict[str, Any]]:
        results = []
        if not raw_results:
            return results
        for idx, item in enumerate(raw_results):
            candidate = None
            if isinstance(item, dict):
                entity_id = item.get("entity_id") or item.get("node_id") or item.get("id")
                if entity_id is None:
                    continue
                entity_id = str(entity_id)
                if entity_id not in self.graph.nodes:
                    continue
                node = self.graph.nodes[entity_id]
                raw_score = item.get("relevance_score", item.get("score", 0.0))
                candidate = {
                    "entity_id": entity_id,
                    "entity_name": item.get("entity_name", getattr(node, "name", "")),
                    "entity_type": item.get("entity_type", getattr(node, "type", "")),
                    "file_path": item.get("file_path", getattr(node, "file_path", "")),
                    "code_snippet": item.get("code_snippet", self._get_node_code(node)),
                    "match_source": item.get("match_source", default_source),
                    "relevance_score": self._normalize_score(raw_score, idx),
                }
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                entity_id = str(item[0])
                if entity_id not in self.graph.nodes:
                    continue
                node = self.graph.nodes[entity_id]
                raw_score = item[1]
                candidate = {
                    "entity_id": entity_id,
                    "entity_name": getattr(node, "name", ""),
                    "entity_type": getattr(node, "type", ""),
                    "file_path": getattr(node, "file_path", ""),
                    "code_snippet": self._get_node_code(node),
                    "match_source": default_source,
                    "relevance_score": self._normalize_score(raw_score, idx),
                }
            else:
                entity_id = str(item)
                if entity_id not in self.graph.nodes:
                    continue
                node = self.graph.nodes[entity_id]
                candidate = {
                    "entity_id": entity_id,
                    "entity_name": getattr(node, "name", ""),
                    "entity_type": getattr(node, "type", ""),
                    "file_path": getattr(node, "file_path", ""),
                    "code_snippet": self._get_node_code(node),
                    "match_source": default_source,
                    "relevance_score": self._normalize_score(None, idx),
                }
            if candidate is not None:
                results.append(candidate)
        dedup = {}
        for c in results:
            key = c["entity_id"]
            if key not in dedup or c["relevance_score"] > dedup[key]["relevance_score"]:
                dedup[key] = c
        final = sorted(dedup.values(), key=lambda x: x["relevance_score"], reverse=True)
        return final[: self.max_candidate_pool]

    def _candidate_from_node(self, node_id: str, base_score: float, source: str) -> Dict[str, Any]:
        node = self.graph.nodes[node_id]
        return {
            "entity_id": str(node_id),
            "entity_name": getattr(node, "name", ""),
            "entity_type": getattr(node, "type", ""),
            "file_path": getattr(node, "file_path", ""),
            "code_snippet": self._get_node_code(node),
            "match_source": source,
            "relevance_score": min(max(float(base_score), 0.0), 1.0),
        }

    def _normalize_score(self, score: Any, rank_idx: int) -> float:
        if score is None:
            return max(0.1, 1.0 - rank_idx * 0.1)
        try:
            val = float(score)
        except Exception:
            return max(0.1, 1.0 - rank_idx * 0.1)
        if 0.0 <= val <= 1.0:
            return val
        if val < 0:
            return 0.0
        return float(1.0 - math.exp(-val / 5.0))

    def _merge_into_candidate_pool(self, new_candidates: List[Dict[str, Any]]):
        merged = {c["entity_id"]: c for c in self.candidate_pool}
        for c in new_candidates:
            key = c["entity_id"]
            if key not in merged:
                merged[key] = c
            else:
                old = merged[key]
                if c.get("relevance_score", 0.0) >= old.get("relevance_score", 0.0):
                    merged[key] = {**old, **c}
        self.candidate_pool = sorted(merged.values(), key=lambda x: x.get("relevance_score", 0.0), reverse=True)[:self.max_candidate_pool]

    def _choose_candidate_after_tool(self, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None
        filtered = [c for c in candidates if c["entity_id"] not in self.history]
        if not filtered:
            filtered = candidates
        if self.reasoner is None:
            selected = filtered[0]
            self.last_reasoner_choice = (selected["entity_id"], selected["relevance_score"])
            self.last_reasoner_trace = {"mode": "no_reasoner_direct_pick"}
            return selected
        reranked = self._try_reasoner_rerank(self.bug_query or "", filtered)
        if reranked:
            selected = reranked[0]
            return selected
        selected = filtered[0]
        self.last_reasoner_choice = (selected["entity_id"], selected["relevance_score"])
        self.last_reasoner_trace = {"mode": "reasoner_failed_fallback"}
        return selected

    def _try_reasoner_rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.reasoner is None:
            return candidates
        try:
            tuple_input = [(c["entity_id"], c["relevance_score"]) for c in candidates]
            reranked = self.reasoner.rerank(query, tuple_input)
            if not reranked:
                return candidates
            score_map = {str(nid): float(score) for nid, score in reranked}
            out = []
            for c in candidates:
                cid = c["entity_id"]
                if cid in score_map:
                    cc = dict(c)
                    cc["relevance_score"] = self._normalize_score(score_map[cid], 0)
                    out.append(cc)
            out = sorted(out, key=lambda x: x["relevance_score"], reverse=True)
            if out:
                self.last_reasoner_choice = (out[0]["entity_id"], out[0]["relevance_score"])
            self.last_reasoner_trace = getattr(self.reasoner, "last_trace", None)
            return out if out else candidates
        except Exception as e:
            self.last_reasoner_choice = None
            self.last_reasoner_trace = {"error": str(e), "mode": "reasoner_exception"}
            return candidates

    def _get_jump_candidates(self) -> List[Dict[str, Any]]:
        pool = list(self.candidate_pool)
        if self.current_node is not None:
            local_neighbors = self._safe_get_neighbors(self.current_node)
            local_candidates = [self._candidate_from_node(nid, 0.2, "neighbor") for nid in local_neighbors]
            for c in local_candidates:
                if not any(x["entity_id"] == c["entity_id"] for x in pool):
                    pool.append(c)
        pool = sorted(pool, key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        return pool[: self.max_jump_candidates]

    def _pick_best_fresh_candidate(self, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for c in sorted(candidates, key=lambda x: x.get("relevance_score", 0.0), reverse=True):
            if c["entity_id"] not in self.visited_set:
                return c
        return candidates[0] if candidates else None

    def _build_submission_candidates(self, submit_topk: int) -> List[Dict[str, Any]]:
        submit_topk = max(1, min(submit_topk, self.top_k_retrieval))
        candidates = []
        seen = set()
        if self.current_node is not None and self.current_node in self.graph.nodes:
            cur = self._candidate_from_node(self.current_node, max(self._get_current_retrieval_score(), 0.5), "current")
            candidates.append(cur)
            seen.add(cur["entity_id"])
        for c in self.candidate_pool:
            cid = c["entity_id"]
            if cid not in seen:
                candidates.append(c)
                seen.add(cid)
            if len(candidates) >= submit_topk:
                break
        return candidates[:submit_topk]

    def _best_candidate_score(self, candidates: List[Dict[str, Any]]) -> float:
        if not candidates:
            return 0.0
        return max(float(c.get("relevance_score", 0.0)) for c in candidates)

    # ------------------------------------------------------------------
    # Verifier
    # ------------------------------------------------------------------

    def _call_verifier(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.verifier is None:
            submitted_ids = [c["entity_id"] for c in candidates]
            if self.bug_node in submitted_ids:
                return {"verdict": self.VERDICT_ACCEPT, "confidence": 0.55, "source": "oracle_fallback"}
            return {"verdict": self.VERDICT_REJECT, "confidence": 0.55, "source": "oracle_fallback"}

        top1 = candidates[0]["entity_id"] if candidates else None
        payload = {
            "query": self.bug_query,
            "candidate_node_id": top1,
            "candidate_node_ids": [c["entity_id"] for c in candidates],
            "candidate_entities": candidates,
            "current_node_id": self.current_node,
            "bug_node_id": self.bug_node,
        }

        if hasattr(self.verifier, "debate"):
            try:
                raw = self.verifier.debate(**payload)
                return self._normalize_verifier_result(raw)
            except TypeError:
                try:
                    raw = self.verifier.debate(query=self.bug_query, candidate_node_id=top1, bug_node_id=self.bug_node)
                    return self._normalize_verifier_result(raw)
                except Exception as e:
                    return {"verdict": self.VERDICT_REJECT, "confidence": 0.2, "error": str(e)}
            except Exception as e:
                return {"verdict": self.VERDICT_REJECT, "confidence": 0.2, "error": str(e)}
        return {"verdict": self.VERDICT_UNKNOWN, "confidence": 0.0, "error": "no_debate_method"}

    def _normalize_verifier_result(self, raw: Any) -> Dict[str, Any]:
        if raw is None:
            return {"verdict": self.VERDICT_UNKNOWN, "confidence": 0.0}
        if isinstance(raw, dict):
            verdict = raw.get("verdict", raw.get("decision", raw.get("label", None)))
            confidence = raw.get("confidence", raw.get("score", 0.0))
            if isinstance(verdict, bool):
                verdict = self.VERDICT_ACCEPT if verdict else self.VERDICT_REJECT
            elif verdict is None and "verdict" in raw and isinstance(raw["verdict"], bool):
                verdict = self.VERDICT_ACCEPT if raw["verdict"] else self.VERDICT_REJECT
            verdict = self._normalize_verdict_label(verdict)
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.0
            out = dict(raw)
            out["verdict"] = verdict
            out["confidence"] = min(max(confidence, 0.0), 1.0)
            return out
        if isinstance(raw, bool):
            return {"verdict": self.VERDICT_ACCEPT if raw else self.VERDICT_REJECT, "confidence": 0.5}
        if isinstance(raw, str):
            return {"verdict": self._normalize_verdict_label(raw), "confidence": 0.5}
        return {"verdict": self.VERDICT_UNKNOWN, "confidence": 0.0, "raw": raw}

    def _normalize_verdict_label(self, verdict: Any) -> str:
        if verdict is None:
            return self.VERDICT_UNKNOWN
        text = str(verdict).strip().lower()
        if text in {"accept", "accepted", "true", "yes", "support"}:
            return self.VERDICT_ACCEPT
        if text in {"reject", "rejected", "false", "no", "oppose"}:
            return self.VERDICT_REJECT
        if text in {"uncertain", "unknown", "maybe", "unsure"}:
            return self.VERDICT_UNCERTAIN
        return self.VERDICT_UNKNOWN

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------

    def _safe_get_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        try:
            if edge_type is None:
                neighbors = self.graph.get_neighbors(node_id)
            else:
                neighbors = self.graph.get_neighbors(node_id, edge_type=edge_type)
        except Exception:
            neighbors = []
        return list(neighbors) if neighbors else []

    def _get_k_hop_nodes(self, start_node: Optional[str], hop_k: int) -> List[str]:
        if start_node is None or start_node not in self.graph.nodes:
            return []
        hop_k = max(1, int(hop_k))
        visited = {start_node}
        queue = deque([(start_node, 0)])
        results = []
        while queue:
            node_id, dist = queue.popleft()
            if dist >= hop_k:
                continue
            for nbr in self._safe_get_neighbors(node_id):
                if nbr in visited:
                    continue
                visited.add(nbr)
                results.append(nbr)
                queue.append((nbr, dist + 1))
        return results

    def _compute_distance(self, start_node: Optional[str], target_node: Optional[str]) -> int:
        if start_node is None or target_node is None:
            return 10
        if start_node == target_node:
            return 0
        visited = set()
        queue = deque([(start_node, 0)])
        while queue:
            node_id, dist = queue.popleft()
            if node_id == target_node:
                return dist
            if node_id in visited:
                continue
            visited.add(node_id)
            for neighbor in self._safe_get_neighbors(node_id):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        return 10

    def _distance_reward(self, old_distance: int, new_distance: int) -> float:
        if new_distance < old_distance:
            return 1.0
        if new_distance == old_distance:
            return 0.0
        return -0.4

    def _visit(self, node_id: str):
        self.history.append(node_id)
        self.visited_set.add(node_id)

    def _action_name(self, action_type: int) -> str:
        return self.ACTION_NAMES.get(int(action_type), "unknown")

    def _tool_name(self, tool_idx: int) -> str:
        return self.TOOL_NAMES.get(int(tool_idx), "unknown_tool")

    def _get_info(self) -> Dict[str, Any]:
        info = {
            "current_node_id": self.current_node,
            "bug_node_id": self.bug_node,
            "bug_query": self.bug_query,
            "query_mode": self.query_mode,
            "steps": self.steps,
            "history_length": len(self.history),
            "visited_count": len(self.visited_set),
            "candidate_pool_size": len(self.candidate_pool),
            "last_retrieval_results": self.last_retrieval_results,
            "last_reasoner_choice": self.last_reasoner_choice,
            "last_reasoner_trace": self.last_reasoner_trace,
            "last_verifier_result": self.last_verifier_result,
            "last_tool_result": self.last_tool_result,
            "last_expand_result": self.last_expand_result,
            "last_submit_candidates": self.last_submit_candidates,
            "action_counter": dict(self.action_counter),
            "tool_counter": dict(self.tool_counter),
            "trajectory_length": len(self.trajectory),
            "reward_breakdown": self.last_reward_breakdown,
        }
        if self.current_node is not None and self.current_node in self.graph.nodes:
            current = self.graph.nodes[self.current_node]
            info.update({
                "current_node_name": getattr(current, "name", ""),
                "current_node_type": getattr(current, "type", ""),
                "current_node_path": getattr(current, "file_path", ""),
            })
        if self.bug_node is not None and self.bug_node in self.graph.nodes:
            bug = self.graph.nodes[self.bug_node]
            info.update({
                "bug_node_name": getattr(bug, "name", ""),
                "bug_node_type": getattr(bug, "type", ""),
                "bug_node_path": getattr(bug, "file_path", ""),
            })
        return info

    def render(self):
        if self.current_node is None or self.bug_node is None:
            print("Environment not reset.")
            return
        current = self.graph.nodes[self.current_node]
        bug = self.graph.nodes[self.bug_node]
        pool_preview = [
            (c.get("entity_name", c.get("entity_id")), round(float(c.get("relevance_score", 0.0)), 3))
            for c in self.candidate_pool[:5]
        ]
        print(f"[Step {self.steps}] Current: {getattr(current, 'name', '')} ({getattr(current, 'type', '')}) | "
              f"Bug: {getattr(bug, 'name', '')} ({getattr(bug, 'type', '')}) | PoolTop5: {pool_preview}")
        if self.last_reward_breakdown:
            print(f"  Reward breakdown: {self.last_reward_breakdown}")

    def _generate_query_from_bug_node(self, bug_node: str) -> str:
        node = self.graph.nodes[bug_node]
        node_name = getattr(node, "name", "")
        node_type = getattr(node, "type", "")
        node_path = getattr(node, "file_path", "")
        node_doc = getattr(node, "doc", "")
        if self.query_mode == "weak":
            query_parts = [node_name, node_type, node_path]
            neighbors = self._safe_get_neighbors(bug_node)
            neighbor_names = [getattr(self.graph.nodes[n], "name", "") for n in neighbors[:5] if n in self.graph.nodes]
            neighbor_names = [x for x in neighbor_names if x]
            if neighbor_names:
                query_parts.append(" ".join(neighbor_names))
            if node_doc:
                query_parts.append(node_doc)
            return " ".join([str(x) for x in query_parts if x]).strip()
        if self.query_mode == "strong":
            query_parts = [node_type]
            if node_path:
                file_basename = node_path.split("/")[-1].replace(".py", "")
                if file_basename:
                    query_parts.append(file_basename)
            if node_doc:
                doc_words = str(node_doc).split()[:8]
                if doc_words:
                    query_parts.append(" ".join(doc_words))
            neighbors = self._safe_get_neighbors(bug_node)
            neighbor_names = []
            for n in neighbors[:3]:
                if n not in self.graph.nodes:
                    continue
                n_name = getattr(self.graph.nodes[n], "name", "")
                if n_name and n_name != node_name:
                    neighbor_names.append(n_name)
            if neighbor_names:
                query_parts.append(" ".join(neighbor_names))
            return " ".join([str(x) for x in query_parts if x]).strip()
        if self.query_mode == "minimal":
            return f"{node_type}"
        query_parts = [node_name, node_type, node_path]
        if node_doc:
            query_parts.append(str(node_doc).split("\n")[0])
        return " ".join([str(x) for x in query_parts if x]).strip()

    def _get_node_code(self, node) -> str:
        code = getattr(node, "code", "")
        if not code:
            return ""
        text = str(code)
        return text[:400]