# reasoner/snippet_refiner.py

import re
from typing import Dict, List, Tuple


class SnippetRefiner:
    """
    轻量级 snippet refinement 模块

    目标：
    1. 在 top-1 function 内做更细粒度定位
    2. 过滤 docstring / 示例 / 注释 / 空行
    3. 优先可执行代码
    4. 使用局部窗口而不是单行做打分
    5. 返回连续 snippet block
    """

    def __init__(self):
        # 与 bug / logic 更相关的关键词
        self.logic_keywords = {
            "if", "elif", "else", "return", "raise", "try", "except",
            "for", "while", "assert", "not", "and", "or",
            "matrix", "shape", "transform", "separable", "inputs",
            "outputs", "compute", "calculate", "nested", "compound",
            "model", "models", "array", "bool", "true", "false",
        }

    # ---------------------------------------------------------
    # Basic text processing
    # ---------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9_]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        return text.split()

    def _token_set(self, text: str) -> set:
        return set(self._tokenize(text))

    # ---------------------------------------------------------
    # Line classification
    # ---------------------------------------------------------

    def _is_blank(self, line: str) -> bool:
        return not line.strip()

    def _is_comment(self, line: str) -> bool:
        return line.lstrip().startswith("#")

    def _is_example_line(self, line: str) -> bool:
        stripped = line.lstrip()
        return stripped.startswith(">>>") or stripped.startswith("...")

    def _looks_like_docstring_text(self, line: str) -> bool:
        """
        粗略识别纯说明性自然语言行。
        不是严格解析 docstring，只是做轻量过滤。
        """
        stripped = line.strip()

        if not stripped:
            return False

        # 三引号本身
        if '"""' in stripped or "'''" in stripped:
            return True

        # docstring 常见标题
        doc_headers = {
            "parameters", "returns", "examples", "notes", "see also",
            "raises", "references", "warning", "warnings"
        }
        low = stripped.lower().strip(":")
        if low in doc_headers:
            return True

        # 典型纯文本行：没有代码操作符，主要是自然语言
        code_markers = ["=", "==", "(", ")", "[", "]", "{", "}", "return", "if ", "for ", "while ", "raise "]
        if not any(m in stripped for m in code_markers):
            # token 大多是字母词，且长度较长，通常像说明文本
            tokens = self._tokenize(stripped)
            if len(tokens) >= 4:
                alpha_tokens = sum(1 for t in tokens if t.isalpha())
                if alpha_tokens / max(len(tokens), 1) >= 0.7:
                    return True

        return False

    def _is_low_value_line(self, line: str) -> bool:
        """
        应尽量过滤掉的行：
        - 空行
        - 注释
        - REPL 示例
        - 看起来像 docstring 说明
        """
        return (
            self._is_blank(line)
            or self._is_comment(line)
            or self._is_example_line(line)
            or self._looks_like_docstring_text(line)
        )

    def _is_executable_code_line(self, line: str) -> bool:
        """
        轻量判断：是否更像真正的执行代码。
        """
        stripped = line.strip()
        if not stripped:
            return False
        if self._is_low_value_line(line):
            return False

        # 常见执行代码特征
        code_patterns = [
            "=", "==", "!=", "<", ">", "<=", ">=",
            "if ", "elif ", "else:", "for ", "while ",
            "return", "raise", "try:", "except", "with ",
            "(", ")", "[", "]", "{", "}",
        ]

        return any(p in stripped for p in code_patterns)

    # ---------------------------------------------------------
    # Scoring
    # ---------------------------------------------------------

    def _keyword_overlap_score(self, query_tokens: set, text: str) -> float:
        line_tokens = self._token_set(text)
        if not query_tokens or not line_tokens:
            return 0.0
        return len(query_tokens & line_tokens) / max(len(query_tokens), 1)

    def _logic_bonus(self, text: str) -> float:
        tokens = self._token_set(text)
        if not tokens:
            return 0.0
        overlap = len(tokens & self.logic_keywords)
        return min(overlap * 0.03, 0.18)

    def _structure_bonus(self, text: str) -> float:
        stripped = text.strip()
        bonus = 0.0

        if self._is_executable_code_line(stripped):
            bonus += 0.08

        for marker in ["if ", "elif ", "return", "raise", "for ", "while ", "="]:
            if marker in stripped:
                bonus += 0.02

        return min(bonus, 0.16)

    def _window_score(self, query_tokens: set, window_lines: List[str]) -> float:
        """
        对 3 行窗口打分。
        """
        joined = "\n".join(window_lines)
        base = self._keyword_overlap_score(query_tokens, joined)
        logic = self._logic_bonus(joined)
        structure = self._structure_bonus(joined)

        return base + logic + structure

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def refine(self, query: str, code: str, top_k: int = 3) -> List[Dict]:
        """
        输入：
            query: bug 描述
            code: top-1 function 的源码
            top_k: 返回多少个 snippet block

        输出：
            [
              {
                "line_start": ...,
                "line_end": ...,
                "content": "...",
                "score": ...
              },
              ...
            ]
        """
        if not code:
            return []

        lines = code.split("\n")
        query_tokens = self._token_set(query)

        if not lines:
            return []

        candidates: List[Dict] = []

        # 以每一行作为中心，构造 3 行窗口
        for i in range(len(lines)):
            start = max(0, i - 1)
            end = min(len(lines), i + 2)  # python slice end exclusive
            window_lines = lines[start:end]

            # 如果整个窗口都是低价值文本，就跳过
            useful_lines = [ln for ln in window_lines if not self._is_low_value_line(ln)]
            if not useful_lines:
                continue

            # 至少有一行像执行代码，才值得作为候选
            if not any(self._is_executable_code_line(ln) for ln in useful_lines):
                continue

            score = self._window_score(query_tokens, useful_lines)
            if score <= 0:
                continue

            content = "\n".join(window_lines).strip()
            if not content:
                continue

            candidates.append({
                "line_start": start + 1,
                "line_end": end,
                "content": content,
                "score": round(score, 4),
            })

        if not candidates:
            return []

        # 去重：按 (line_start, line_end)
        dedup = {}
        for c in candidates:
            key = (c["line_start"], c["line_end"])
            if key not in dedup or c["score"] > dedup[key]["score"]:
                dedup[key] = c

        candidates = list(dedup.values())
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # 避免高度重叠的窗口全返回
        selected = []
        used_ranges: List[Tuple[int, int]] = []

        for cand in candidates:
            s1, e1 = cand["line_start"], cand["line_end"]

            overlap_too_much = False
            for s2, e2 in used_ranges:
                overlap = max(0, min(e1, e2) - max(s1, s2) + 1)
                shorter = min(e1 - s1 + 1, e2 - s2 + 1)
                if shorter > 0 and overlap / shorter >= 0.67:
                    overlap_too_much = True
                    break

            if overlap_too_much:
                continue

            selected.append(cand)
            used_ranges.append((s1, e1))

            if len(selected) >= top_k:
                break

        return selected