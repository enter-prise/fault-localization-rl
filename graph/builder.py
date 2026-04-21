# graph/builder.py

import ast
import os
from typing import Dict, List, Optional, Tuple

from graph.schema import CodeGraph, Node, Edge


class FunctionCallVisitor(ast.NodeVisitor):
    """
    收集函数/方法体内部的：
    1. 调用名 calls
    2. import 信息 imports
    """
    def __init__(self):
        self.calls: List[str] = []
        self.imports: List[str] = []

    def visit_Call(self, node):
        call_name = self._extract_call_name(node.func)
        if call_name:
            self.calls.append(call_name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            # import os / import numpy as np
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module or ""
        for alias in node.names:
            # from x import y  -> x.y
            full_name = f"{module}.{alias.name}" if module else alias.name
            self.imports.append(full_name)
        self.generic_visit(node)

    def _extract_call_name(self, func_node) -> Optional[str]:
        """
        尽量提取调用目标名：
        foo()              -> foo
        obj.bar()          -> bar
        pkg.mod.func()     -> func
        """
        if isinstance(func_node, ast.Name):
            return func_node.id

        if isinstance(func_node, ast.Attribute):
            return func_node.attr

        return None


class GraphBuilder:
    def __init__(self):
        self.node_id = 0
        # 更细粒度索引
        self.function_id_to_file: Dict[str, str] = {}
        self.function_id_to_class: Dict[str, Optional[str]] = {}
        self.function_key_to_id: Dict[Tuple[str, str, Optional[str]], str] = {}
        self.function_name_to_ids: Dict[str, List[str]] = {}
        self.class_name_to_ids: Dict[str, List[str]] = {}
        self.file_path_to_id: Dict[str, str] = {}

        # 延迟解析的边
        self.pending_calls: List[Tuple[str, str]] = []  # (caller_func_id, callee_name)
        self.pending_imports: List[Tuple[str, str]] = []  # (file_id, import_name)

    def _new_id(self) -> str:
        self.node_id += 1
        return str(self.node_id)

    def build_from_repo(self, repo_path: str) -> CodeGraph:
        """
        从代码仓库中构建图。
        当前版本支持：
        - file / class / function / bug 节点
        - contains / imports / calls / related_to / affects 边
        """
        graph = CodeGraph()

        for root, dirs, files in os.walk(repo_path):
            # 跳过缓存、虚拟环境、git 等
            dirs[:] = [
                d for d in dirs
                if d not in {".git", "__pycache__", ".venv", "venv", "env", "node_modules"}
            ]

            for file_name in files:
                if not file_name.endswith(".py"):
                    continue

                abs_file_path = os.path.join(root, file_name)
                rel_file_path = os.path.relpath(abs_file_path, repo_path)

                source = self._safe_read_file(abs_file_path)
                if source is None:
                    continue

                # 1) 创建 file 节点
                file_id = self._new_id()
                file_node = Node(
                    node_id=file_id,
                    name=file_name,
                    node_type="file",
                    code=source,
                    doc="",
                    file_path=rel_file_path,
                )
                graph.add_node(file_node)
                self.file_path_to_id[rel_file_path] = file_id

                # 2) 解析 AST
                try:
                    tree = ast.parse(source, filename=rel_file_path)
                except SyntaxError:
                    continue

                # 3) 提取模块级 import
                module_imports = self._extract_module_imports(tree)
                for import_name in module_imports:
                    self.pending_imports.append((file_id, import_name))

                # 4) 提取顶层 class / function
                self._process_module_body(
                    graph=graph,
                    body=tree.body,
                    file_id=file_id,
                    rel_file_path=rel_file_path,
                    source=source,
                    parent_class_id=None,
                )

                # 5) 处理与该文件相关的 bug report
                self._attach_bug_reports(graph, rel_file_path, file_id)

        # 第二阶段：补全延迟边
        self._resolve_pending_calls(graph)
        self._resolve_pending_imports(graph)

        return graph

    def _safe_read_file(self, file_path: str) -> Optional[str]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read()
            except Exception:
                return None
        except Exception:
            return None

    def _extract_module_imports(self, tree: ast.AST) -> List[str]:
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)
        return imports

    def _process_module_body(
        self,
        graph: CodeGraph,
        body: List[ast.stmt],
        file_id: str,
        rel_file_path: str,
        source: str,
        parent_class_id: Optional[str] = None,
    ):
        """
        处理一个作用域下的 body：
        - 顶层模块 body
        - class body
        """
        for node in body:
            if isinstance(node, ast.ClassDef):
                self._add_class_node(
                    graph=graph,
                    class_node=node,
                    file_id=file_id,
                    rel_file_path=rel_file_path,
                    source=source,
                )

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._add_function_node(
                    graph=graph,
                    func_node=node,
                    file_id=file_id,
                    rel_file_path=rel_file_path,
                    source=source,
                    parent_class_id=parent_class_id,
                )

    def _add_class_node(
        self,
        graph: CodeGraph,
        class_node: ast.ClassDef,
        file_id: str,
        rel_file_path: str,
        source: str,
    ):
        class_id = self._new_id()
        class_code = self._get_node_source_segment(source, class_node)
        class_doc = ast.get_docstring(class_node) or ""

        node = Node(
            node_id=class_id,
            name=class_node.name,
            node_type="class",
            code=class_code,
            doc=class_doc,
            file_path=rel_file_path,
        )
        graph.add_node(node)
        graph.add_edge(Edge(file_id, class_id, "contains"))

        self.class_name_to_ids.setdefault(class_node.name, []).append(class_id)

        for item in class_node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._add_function_node(
                    graph=graph,
                    func_node=item,
                    file_id=file_id,
                    rel_file_path=rel_file_path,
                    source=source,
                    parent_class_id=class_id,
                )

    def _add_function_node(
        self,
        graph: CodeGraph,
        func_node,
        file_id: str,
        rel_file_path: str,
        source: str,
        parent_class_id: Optional[str] = None,
    ):
        func_id = self._new_id()

        func_code = self._get_node_source_segment(source, func_node)
        func_doc = ast.get_docstring(func_node) or ""

        node = Node(
            node_id=func_id,
            name=func_node.name,
            node_type="function",
            code=func_code,
            doc=func_doc,
            file_path=rel_file_path,
        )
        graph.add_node(node)

        if parent_class_id is not None:
            graph.add_edge(Edge(parent_class_id, func_id, "contains"))
        else:
            graph.add_edge(Edge(file_id, func_id, "contains"))

        self.function_name_to_ids.setdefault(func_node.name, []).append(func_id)

        self.function_id_to_file[func_id] = rel_file_path
        self.function_id_to_class[func_id] = parent_class_id
        self.function_key_to_id[(rel_file_path, func_node.name, parent_class_id)] = func_id

        visitor = FunctionCallVisitor()
        visitor.visit(func_node)

        for callee_name in visitor.calls:
            self.pending_calls.append((func_id, callee_name))

        for import_name in visitor.imports:
            self.pending_imports.append((file_id, import_name))

    def _get_node_source_segment(self, source: str, node: ast.AST) -> str:
        try:
            segment = ast.get_source_segment(source, node)
            return segment or ""
        except Exception:
            return ""
        
    def _resolve_pending_calls(self, graph: CodeGraph):
        for caller_id, callee_name in self.pending_calls:
            caller_file = self.function_id_to_file.get(caller_id)
            caller_class = self.function_id_to_class.get(caller_id)

            matched_target_ids = []

            if caller_file is not None and caller_class is not None:
                target_id = self.function_key_to_id.get((caller_file, callee_name, caller_class))
                if target_id and target_id != caller_id:
                    matched_target_ids.append(target_id)

            if not matched_target_ids and caller_file is not None:
                target_id = self.function_key_to_id.get((caller_file, callee_name, None))
                if target_id and target_id != caller_id:
                    matched_target_ids.append(target_id)

            if not matched_target_ids:
                global_matches = self.function_name_to_ids.get(callee_name, [])
                global_matches = [fid for fid in global_matches if fid != caller_id]

                if len(global_matches) == 1:
                    matched_target_ids.extend(global_matches)

            for target_id in matched_target_ids:
                graph.add_edge(Edge(caller_id, target_id, "calls"))

    def _resolve_pending_imports(self, graph: CodeGraph):
        file_name_to_id = {}
        for node_id, node in graph.nodes.items():
            if node.type == "file":
                base_name = os.path.splitext(os.path.basename(node.file_path))[0]
                file_name_to_id.setdefault(base_name, []).append(node_id)

        for src_file_id, import_name in self.pending_imports:
            last_part = import_name.split(".")[-1]
            target_file_ids = file_name_to_id.get(last_part, [])
            for dst_file_id in target_file_ids:
                if src_file_id != dst_file_id:
                    graph.add_edge(Edge(src_file_id, dst_file_id, "imports"))

    def _attach_bug_reports(self, graph: CodeGraph, rel_file_path: str, file_id: str):
        bug_reports = self.get_bug_reports(rel_file_path)
        if not bug_reports:
            return

        direct_children = graph.get_neighbors(file_id)

        for bug in bug_reports:
            bug_desc = bug.get("description", "").strip()
            if not bug_desc:
                continue

            bug_id = self._new_id()
            bug_node = Node(
                node_id=bug_id,
                name=bug_desc[:80],
                node_type="bug",
                code="",
                doc=bug_desc,
                file_path=rel_file_path,
            )
            graph.add_node(bug_node)

            graph.add_edge(Edge(file_id, bug_id, "related_to"))

            for child_id in direct_children:
                child = graph.nodes.get(child_id)
                if child and child.type in {"function", "class"}:
                    graph.add_edge(Edge(child_id, bug_id, "affects"))

    def get_bug_reports(self, rel_file_path: str):
        bug_reports = []

        base_name = os.path.basename(rel_file_path)
        candidate_paths = [
            os.path.join("bug_reports", f"{base_name}.txt"),
            os.path.join("bug_reports", f"{rel_file_path}.txt"),
        ]

        for bug_file_path in candidate_paths:
            if os.path.exists(bug_file_path):
                try:
                    with open(bug_file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                bug_reports.append({"description": line})
                except Exception:
                    pass

        return bug_reports