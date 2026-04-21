import math
import random
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FaultLocalizationEnv(gym.Env):
    """
    更贴近论文结构的 Fault Localization RL Environment。

    核心变化：
    1. 动作从 3 类改为 4 类：JUMP_TO / CALL(tool) / EXPAND(subgraph) / SUBMIT(result)
    2. observation 从简单 14 维统计量扩展为更丰富的状态代理向量，尽量对应：
       v_t（当前节点）、g_t（局部子图）、h_t（历史轨迹）、q_t（issue/query）
    3. 增加候选池、工具缓存、轨迹日志，便于后续 PPO 和 verifier 闭环学习
    4. 尽量兼容你现有 graph / retriever / reasoner / verifier 接口

    说明：
    - 这是“论文骨架版环境”，不是最终完全实验版。
    - 仍然允许用 oracle bug_node 做奖励塑形，但 observation 不显式泄露 bug_node。
    - 如果你的 retriever / verifier 接口后续升级，这个 env 可以继续扩展。
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
        max_steps: int = 20,
        top_k_retrieval: int = 5,
        max_jump_candidates: int = 8,
        max_expand_hop: int = 3,
        max_candidate_pool: int = 20,
        seed: Optional[int] = None,
        query_mode: str = "weak",
        reasoner=None,
        verifier=None,
        auto_terminate_on_exact_hit: bool = False,
    ):
        super().__init__()

        self.graph = graph
        self.retriever = retriever
        self.reasoner = reasoner
        self.verifier = verifier

        self.query_mode = query_mode
        self.user_bug_query = bug_query
        self.fixed_bug_node = bug_node

        self.max_steps = max_steps
        self.top_k_retrieval = max(1, int(top_k_retrieval))
        self.max_jump_candidates = max(1, int(max_jump_candidates))
        self.max_expand_hop = max(1, int(max_expand_hop))
        self.max_candidate_pool = max(self.top_k_retrieval, int(max_candidate_pool))
        self.auto_terminate_on_exact_hit = auto_terminate_on_exact_hit

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

        # MultiDiscrete 设计：
        # [action_type, jump_idx, tool_idx, expand_hop_raw, submit_topk_raw]
        # - action_type: 0 jump / 1 call / 2 expand / 3 submit
        # - jump_idx: 在 candidate_pool 中选择第几个候选
        # - tool_idx: 选择哪个工具
        # - expand_hop_raw: 0..max_expand_hop-1，对应 hop=1..max_expand_hop
        # - submit_topk_raw: 0..top_k_retrieval-1，对应 topk=1..top_k_retrieval
        self.action_space = spaces.MultiDiscrete([
            4,
            self.max_jump_candidates,
            3,
            self.max_expand_hop,
            self.top_k_retrieval,
        ])

        # 32 维状态代理向量：
        # 0-5   当前节点类型 one-hot（repo/dir/file/class/function/other）
        # 6-10  当前节点基本属性/分数
        # 11-15 当前 1-hop 邻居统计
        # 16-19 当前 k-hop 子图统计
        # 20-23 历史行为比例
        # 24-27 verifier / tool / candidate pool 状态
        # 28-31 query 与 episode 进度
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(32,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Reset / sampling
    # ------------------------------------------------------------------

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

        self._sample_bug_case()

        self.current_node = self.rng.choice(self.node_ids)
        self._visit(self.current_node)

        # reset 时先做一次初始 candidate generation，更接近论文里的 initial candidates
        self._bootstrap_candidate_pool()

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

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

        old_distance = self._compute_distance(self.current_node, self.bug_node)
        reward = -0.05  # 每步轻微成本，鼓励少走弯路
        terminated = False
        truncated = False

        before_node = self.current_node

        if action_type == self.ACTION_JUMP:
            delta_reward, action_detail = self._handle_jump(jump_idx, old_distance)
            reward += delta_reward

        elif action_type == self.ACTION_CALL:
            delta_reward, action_detail = self._handle_call(tool_idx, old_distance)
            reward += delta_reward

        elif action_type == self.ACTION_EXPAND:
            delta_reward, action_detail = self._handle_expand(expand_hop, old_distance)
            reward += delta_reward

        elif action_type == self.ACTION_SUBMIT:
            delta_reward, terminated, action_detail = self._handle_submit(submit_topk)
            reward += delta_reward

        else:
            reward -= 1.0
            action_detail = {"error": "invalid_action_type"}

        if self.current_node is not None:
            self._visit(self.current_node)

        # 到达真实 bug 节点只给中间奖励，默认不立刻终止；更符合“需要 SUBMIT 才结束”的论文语义
        if self.current_node == self.bug_node and not terminated:
            reward += 1.5
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
            "tool_name": self._tool_name(tool_idx) if action_type == self.ACTION_CALL else None,
            "expand_hop": expand_hop if action_type == self.ACTION_EXPAND else None,
            "submit_topk": submit_topk if action_type == self.ACTION_SUBMIT else None,
            "reward": reward,
            "detail": action_detail,
        }
        self.trajectory.append(transition)

        obs = self._get_observation()
        info = self._get_info()
        info["transition"] = transition
        info["reward"] = reward
        info["terminated"] = terminated
        info["truncated"] = truncated
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_jump(self, jump_idx: int, old_distance: int) -> Tuple[float, Dict[str, Any]]:
        jump_candidates = self._get_jump_candidates()
        if not jump_candidates:
            return -0.6, {"status": "no_jump_candidates"}

        selected = jump_candidates[min(jump_idx, len(jump_candidates) - 1)]
        target_node = selected["entity_id"]
        if target_node not in self.graph.nodes:
            return -0.6, {"status": "invalid_target", "selected": selected}

        self.current_node = target_node
        new_distance = self._compute_distance(self.current_node, self.bug_node)

        novelty_bonus = 0.3 if target_node not in self.history else -0.2
        relevance_bonus = 0.35 * min(float(selected.get("relevance_score", 0.0)), 1.0)
        distance_bonus = self._distance_reward(old_distance, new_distance)

        detail = {
            "status": "ok",
            "selected": selected,
            "old_distance": old_distance,
            "new_distance": new_distance,
        }
        return distance_bonus + novelty_bonus + relevance_bonus, detail

    def _handle_call(self, tool_idx: int, old_distance: int) -> Tuple[float, Dict[str, Any]]:
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
            return -0.8, {"status": "unknown_tool", "tool_idx": tool_idx}

        self.last_tool_result = {
            "tool_idx": tool_idx,
            "tool_name": tool_name,
            "results": results,
        }

        if not results:
            return -0.5, {
                "status": "empty_results",
                "tool_name": tool_name,
            }

        self.last_retrieval_results = results
        self._merge_into_candidate_pool(results)

        chosen = self._choose_candidate_after_tool(results)
        if chosen is None:
            return -0.4, {
                "status": "no_chosen_candidate",
                "tool_name": tool_name,
                "results": results,
            }

        self.current_node = chosen["entity_id"]
        new_distance = self._compute_distance(self.current_node, self.bug_node)

        score_bonus = 0.4 * min(float(chosen.get("relevance_score", 0.0)), 1.0)
        distance_bonus = self._distance_reward(old_distance, new_distance)
        novelty_bonus = 0.25 if self.current_node not in self.history else -0.1

        # 工具调用代价：如果过度重复调用同一工具，轻微惩罚
        overuse_penalty = 0.0
        if self.tool_counter[tool_name] >= 3:
            overuse_penalty -= 0.1 * min(self.tool_counter[tool_name] - 2, 3)

        detail = {
            "status": "ok",
            "tool_name": tool_name,
            "chosen": chosen,
            "result_count": len(results),
            "old_distance": old_distance,
            "new_distance": new_distance,
            "reasoner_choice": self.last_reasoner_choice,
            "reasoner_trace": self.last_reasoner_trace,
        }
        return score_bonus + distance_bonus + novelty_bonus + overuse_penalty, detail

    def _handle_expand(self, hop_k: int, old_distance: int) -> Tuple[float, Dict[str, Any]]:
        if self.current_node is None:
            return -0.6, {"status": "no_current_node"}

        nodes_in_subgraph = self._get_k_hop_nodes(self.current_node, hop_k)
        if not nodes_in_subgraph:
            return -0.4, {"status": "empty_subgraph", "hop_k": hop_k}

        expanded_candidates = []
        for nid in nodes_in_subgraph:
            if nid not in self.graph.nodes:
                continue
            expanded_candidates.append(self._candidate_from_node(nid, 0.25, "expand"))

        self._merge_into_candidate_pool(expanded_candidates)

        self.last_expand_result = {
            "center_node": self.current_node,
            "hop_k": hop_k,
            "subgraph_size": len(nodes_in_subgraph),
            "candidates_added": len(expanded_candidates),
        }

        # expand 不一定立刻换 current node；但如果池里出现更优未访问候选，可以轻微跳转到它
        best_fresh = self._pick_best_fresh_candidate(self.candidate_pool)
        jumped = None
        if best_fresh is not None and best_fresh["entity_id"] != self.current_node:
            jumped = best_fresh
            self.current_node = best_fresh["entity_id"]

        new_distance = self._compute_distance(self.current_node, self.bug_node)
        coverage_bonus = min(len(nodes_in_subgraph) / 15.0, 1.0) * 0.5
        novelty_bonus = min(sum(1 for nid in nodes_in_subgraph if nid not in self.visited_set) / 10.0, 1.0) * 0.4
        distance_bonus = self._distance_reward(old_distance, new_distance)

        detail = {
            "status": "ok",
            "hop_k": hop_k,
            "subgraph_size": len(nodes_in_subgraph),
            "jumped_to": jumped,
            "old_distance": old_distance,
            "new_distance": new_distance,
        }
        return coverage_bonus + novelty_bonus + distance_bonus, detail

    def _handle_submit(self, submit_topk: int) -> Tuple[float, bool, Dict[str, Any]]:
        candidates = self._build_submission_candidates(submit_topk)
        self.last_submit_candidates = candidates

        if not candidates:
            self.last_verifier_result = {
                "verdict": self.VERDICT_REJECT,
                "confidence": 0.0,
                "error": "empty_submission_candidates",
            }
            return -4.0, True, {"status": "empty_submission"}

        submitted_ids = [c["entity_id"] for c in candidates]
        top1 = submitted_ids[0]
        hit_top1 = top1 == self.bug_node
        hit_topk = self.bug_node in submitted_ids

        verifier_bonus = 0.0
        verifier_result = self._call_verifier(candidates)
        self.last_verifier_result = verifier_result

        verdict = verifier_result.get("verdict", self.VERDICT_UNKNOWN)
        confidence = float(verifier_result.get("confidence", 0.0))
        confidence = min(max(confidence, 0.0), 1.0)

        if verdict == self.VERDICT_ACCEPT:
            verifier_bonus += 0.8 + 0.4 * confidence
        elif verdict == self.VERDICT_REJECT:
            verifier_bonus -= 0.8 + 0.4 * confidence
        elif verdict == self.VERDICT_UNCERTAIN:
            verifier_bonus += 0.1 * confidence
        else:
            verifier_bonus -= 0.1

        # final reward: 论文里 SUBMIT(result) 触发最终判断，更强调 top-k 提交质量
        if hit_top1:
            final_reward = 10.0
        elif hit_topk:
            final_reward = 6.0
        else:
            final_reward = -5.0

        # 提交候选过多会稀释结果，轻微惩罚
        size_penalty = -0.1 * max(len(candidates) - 3, 0)

        detail = {
            "status": "ok",
            "submitted_ids": submitted_ids,
            "hit_top1": hit_top1,
            "hit_topk": hit_topk,
            "verifier_result": verifier_result,
        }
        return final_reward + verifier_bonus + size_penalty, True, detail

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        if self.current_node is None:
            return np.zeros((32,), dtype=np.float32)

        node = self.graph.nodes[self.current_node]
        node_type = getattr(node, "type", "")
        node_name = str(getattr(node, "name", "") or "")
        node_path = str(getattr(node, "file_path", "") or "")
        node_doc = str(getattr(node, "doc", "") or "")

        # 当前节点类型 one-hot 编码
        type_repo = 1.0 if node_type in {"repository", "repo"} else 0.0
        type_dir = 1.0 if node_type in {"directory", "dir"} else 0.0
        type_file = 1.0 if node_type == "file" else 0.0
        type_class = 1.0 if node_type == "class" else 0.0
        type_function = 1.0 if node_type in {"function", "method"} else 0.0
        known_type = max(type_repo, type_dir, type_file, type_class, type_function)
        type_other = 1.0 - known_type

        # 当前节点基本属性/分数
        name_len_norm = min(len(node_name) / 64.0, 1.0)
        path_len_norm = min(len(node_path) / 128.0, 1.0)
        doc_len_norm = min(len(node_doc.split()) / 80.0, 1.0)
        current_relevance_norm = min(self._get_current_retrieval_score(), 1.0)
        revisit_flag = 1.0 if self.history.count(self.current_node) > 1 else 0.0

        # 1-hop 邻居统计
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

        # k-hop 子图统计（对应 g_t 的代理）
        k2_nodes = self._get_k_hop_nodes(self.current_node, 2)
        k2_size_norm = min(len(k2_nodes) / 30.0, 1.0)
        k2_unvisited_ratio = 0.0
        if k2_nodes:
            k2_unvisited_ratio = min(sum(1 for nid in k2_nodes if nid not in self.visited_set) / max(len(k2_nodes), 1), 1.0)
        candidate_pool_size_norm = min(len(self.candidate_pool) / max(self.max_candidate_pool, 1), 1.0)
        best_pool_score_norm = min(self._best_candidate_score(self.candidate_pool), 1.0)

        # 历史行为比例（对应 h_t 的代理）
        jump_ratio = self.action_counter[self._action_name(self.ACTION_JUMP)] / max(self.steps, 1)
        call_ratio = self.action_counter[self._action_name(self.ACTION_CALL)] / max(self.steps, 1)
        expand_ratio = self.action_counter[self._action_name(self.ACTION_EXPAND)] / max(self.steps, 1)
        submit_ratio = self.action_counter[self._action_name(self.ACTION_SUBMIT)] / max(self.steps, 1)

        # verifier / tool / pool 状态
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

        # query / episode 信息（对应 q_t 的代理）
        query_len_norm = min(len((self.bug_query or "").split()) / 40.0, 1.0)
        step_ratio = self.steps / max(self.max_steps, 1)
        visited_ratio = min(len(self.visited_set) / max(len(self.node_ids), 1), 1.0)
        current_in_pool = 1.0 if any(c["entity_id"] == self.current_node for c in self.candidate_pool) else 0.0

        # 将所有特征拼接成 32 维的 observation
        observation = np.array([
            type_repo,
            type_dir,
            type_file,
            type_class,
            type_function,
            type_other,
            name_len_norm,
            path_len_norm,
            doc_len_norm,
            current_relevance_norm,
            revisit_flag,
            total_neighbors_norm,
            contains_neighbors_norm,
            calls_neighbors_norm,
            imports_neighbors_norm,
            other_neighbors_norm,
            k2_size_norm,
            k2_unvisited_ratio,
            candidate_pool_size_norm,
            best_pool_score_norm,
            jump_ratio,
            call_ratio,
            expand_ratio,
            submit_ratio,
            verdict_accept,
            verdict_reject,
            verdict_uncertain,
            verdict_confidence,
            query_len_norm,
            step_ratio,
            visited_ratio,
            current_in_pool,
        ], dtype=np.float32)

        # 确保返回的是 32 维的 obs
        obs = observation[:32]  # 如果超过 32 维，就截断
        return obs

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

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------

    def _parse_action(self, action) -> Dict[str, int]:
        # 兼容旧版离散动作：
        # 0 -> jump_to(best)
        # 1 -> call_tool(semantic_scout)
        # 2 -> expand(1-hop)
        # 3 -> submit(top-1)
        if isinstance(action, (int, np.integer)):
            action = int(action)
            if action == 0:
                return {
                    "action_type": self.ACTION_JUMP,
                    "jump_idx": 0,
                    "tool_idx": self.TOOL_SEMANTIC_SCOUT,
                    "expand_hop": 1,
                    "submit_topk": 1,
                }
            if action == 1:
                return {
                    "action_type": self.ACTION_CALL,
                    "jump_idx": 0,
                    "tool_idx": self.TOOL_SEMANTIC_SCOUT,
                    "expand_hop": 1,
                    "submit_topk": 1,
                }
            if action == 2:
                return {
                    "action_type": self.ACTION_EXPAND,
                    "jump_idx": 0,
                    "tool_idx": self.TOOL_SEMANTIC_SCOUT,
                    "expand_hop": 1,
                    "submit_topk": 1,
                }
            if action == 3:
                return {
                    "action_type": self.ACTION_SUBMIT,
                    "jump_idx": 0,
                    "tool_idx": self.TOOL_SEMANTIC_SCOUT,
                    "expand_hop": 1,
                    "submit_topk": 1,
                }

        # MultiDiscrete
        if isinstance(action, np.ndarray):
            action = action.tolist()

        if not isinstance(action, (list, tuple)) or len(action) < 5:
            raise ValueError(
                "Action must be int (legacy) or list/tuple/ndarray of length 5: "
                "[action_type, jump_idx, tool_idx, expand_hop_raw, submit_topk_raw]"
            )

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

    # ------------------------------------------------------------------
    # Candidate pool / tools
    # ------------------------------------------------------------------

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

        # 如果有 reasoner，则在局部邻域上 rerank
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
        # 兼容多种 retriever 接口
        # 1) hierarchical_retrieve(query, top_k=...)
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

        # 2) retrieve_with_score(query, top_k=...)
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

        # 3) retrieve(query, top_k=...)
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
        results: List[Dict[str, Any]] = []
        if not raw_results:
            return results

        for idx, item in enumerate(raw_results):
            candidate = None

            # 情况 1: dict
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

            # 情况 2: tuple/list -> (node_id, score)
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

            # 情况 3: 只有 node_id
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

        # 去重并按分数排序
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
            # 没有 score 时给一个基于排名的衰减分
            return max(0.1, 1.0 - rank_idx * 0.1)

        try:
            val = float(score)
        except Exception:
            return max(0.1, 1.0 - rank_idx * 0.1)

        # 常见情况：
        # - 已经在 [0,1]
        # - BM25 / raw score 大于 1
        if 0.0 <= val <= 1.0:
            return val
        if val < 0:
            return 0.0

        # 用平滑函数压到 [0,1)
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
                    # 保留更高分，同时记录更强来源
                    merged[key] = {
                        **old,
                        **c,
                    }
        self.candidate_pool = sorted(
            merged.values(),
            key=lambda x: x.get("relevance_score", 0.0),
            reverse=True,
        )[: self.max_candidate_pool]

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
        # 优先从 candidate_pool 里拿，补充当前邻居
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

        # 当前节点应优先参与提交
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
            # 无 verifier 时给一个默认输出，便于流程跑通
            submitted_ids = [c["entity_id"] for c in candidates]
            if self.bug_node in submitted_ids:
                return {
                    "verdict": self.VERDICT_ACCEPT,
                    "confidence": 0.55,
                    "source": "oracle_fallback_no_verifier",
                }
            return {
                "verdict": self.VERDICT_REJECT,
                "confidence": 0.55,
                "source": "oracle_fallback_no_verifier",
            }

        top1 = candidates[0]["entity_id"] if candidates else None
        payload = {
            "query": self.bug_query,
            "candidate_node_id": top1,
            "candidate_node_ids": [c["entity_id"] for c in candidates],
            "candidate_entities": candidates,
            "current_node_id": self.current_node,
            "bug_node_id": self.bug_node,
        }

        # 兼容旧版 verifier.debate(query, candidate_node_id, bug_node_id)
        if hasattr(self.verifier, "debate"):
            try:
                raw = self.verifier.debate(**payload)
                return self._normalize_verifier_result(raw)
            except TypeError:
                try:
                    raw = self.verifier.debate(
                        query=self.bug_query,
                        candidate_node_id=top1,
                        bug_node_id=self.bug_node,
                    )
                    return self._normalize_verifier_result(raw)
                except Exception as e:
                    return {
                        "verdict": self.VERDICT_REJECT,
                        "confidence": 0.2,
                        "error": str(e),
                    }
            except Exception as e:
                return {
                    "verdict": self.VERDICT_REJECT,
                    "confidence": 0.2,
                    "error": str(e),
                }

        return {
            "verdict": self.VERDICT_UNKNOWN,
            "confidence": 0.0,
            "error": "verifier_has_no_debate_method",
        }

    def _normalize_verifier_result(self, raw: Any) -> Dict[str, Any]:
        if raw is None:
            return {
                "verdict": self.VERDICT_UNKNOWN,
                "confidence": 0.0,
                "source": "none",
            }

        # dict 格式最常见
        if isinstance(raw, dict):
            verdict = raw.get("verdict", raw.get("decision", raw.get("label", None)))
            confidence = raw.get("confidence", raw.get("score", 0.0))

            # 兼容旧 bool verdict
            if isinstance(verdict, bool):
                verdict = self.VERDICT_ACCEPT if verdict else self.VERDICT_REJECT
            elif verdict is None and "verdict" in raw and isinstance(raw["verdict"], bool):
                verdict = self.VERDICT_ACCEPT if raw["verdict"] else self.VERDICT_REJECT
            elif verdict is None and isinstance(raw.get("support_score", None), (int, float)) and isinstance(raw.get("oppose_score", None), (int, float)):
                support = float(raw.get("support_score", 0.0))
                oppose = float(raw.get("oppose_score", 0.0))
                diff = support - oppose
                if diff > 0.2:
                    verdict = self.VERDICT_ACCEPT
                elif diff < -0.2:
                    verdict = self.VERDICT_REJECT
                else:
                    verdict = self.VERDICT_UNCERTAIN
                confidence = min(abs(diff), 1.0)

            verdict = self._normalize_verdict_label(verdict)
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.0

            out = dict(raw)
            out["verdict"] = verdict
            out["confidence"] = min(max(confidence, 0.0), 1.0)
            return out

        # bool 直接转 accept/reject
        if isinstance(raw, bool):
            return {
                "verdict": self.VERDICT_ACCEPT if raw else self.VERDICT_REJECT,
                "confidence": 0.5,
                "source": "bool_cast",
            }

        # str 直接标准化
        if isinstance(raw, str):
            return {
                "verdict": self._normalize_verdict_label(raw),
                "confidence": 0.5,
                "source": "str_cast",
            }

        return {
            "verdict": self.VERDICT_UNKNOWN,
            "confidence": 0.0,
            "raw": raw,
        }

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
    # Graph helpers / distance
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

    # ------------------------------------------------------------------
    # Info / render
    # ------------------------------------------------------------------

    def _action_name(self, action_type: int) -> str:
        return self.ACTION_NAMES.get(int(action_type), "unknown")

    def _tool_name(self, tool_idx: int) -> str:
        return self.TOOL_NAMES.get(int(tool_idx), "unknown_tool")

    def _get_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
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

        print(
            f"[Step {self.steps}] "
            f"Current: {getattr(current, 'name', '')} ({getattr(current, 'type', '')}) | "
            f"Bug: {getattr(bug, 'name', '')} ({getattr(bug, 'type', '')}) | "
            f"PoolTop5: {pool_preview}"
        )

    # ------------------------------------------------------------------
    # Query generation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _get_node_code(self, node) -> str:
        code = getattr(node, "code", "")
        if not code:
            return ""
        text = str(code)
        return text[:400]
