import os
import random
from collections import deque

import matplotlib.pyplot as plt
import networkx as nx

from graph.builder import GraphBuilder


def extract_subgraph_bfs(graph, start_node_id, max_depth=2, max_nodes=60, edge_types=None):
    """
    从 start_node_id 出发，用 BFS 抽取局部子图。
    edge_types:
        None -> 不限制边类型
        {"calls", "contains", "imports"} -> 只走指定类型边
    """
    visited = set()
    queue = deque([(start_node_id, 0)])

    while queue and len(visited) < max_nodes:
        current, depth = queue.popleft()

        if current in visited:
            continue

        visited.add(current)

        if depth >= max_depth:
            continue

        for edge in graph.get_out_edges(current):
            if edge_types is not None and edge.type not in edge_types:
                continue
            if edge.dst not in visited:
                queue.append((edge.dst, depth + 1))

    return visited


def build_networkx_subgraph(graph, node_ids, edge_types=None):
    G = nx.DiGraph()

    for node_id in node_ids:
        node = graph.nodes[node_id]
        short_name = node.name if len(node.name) <= 28 else node.name[:25] + "..."
        label = f"{short_name}\n[{node.type}]"
        G.add_node(
            node_id,
            label=label,
            node_type=node.type,
            file_path=node.file_path
        )

    for edge in graph.edges:
        if edge.src in node_ids and edge.dst in node_ids:
            if edge_types is not None and edge.type not in edge_types:
                continue
            G.add_edge(edge.src, edge.dst, edge_type=edge.type)

    return G


def get_node_colors(G):
    colors = []
    for _, data in G.nodes(data=True):
        node_type = data.get("node_type", "unknown")
        if node_type == "file":
            colors.append("#8ecae6")   # 浅蓝
        elif node_type == "class":
            colors.append("#ffb703")   # 橙黄
        elif node_type == "function":
            colors.append("#90be6d")   # 浅绿
        elif node_type == "bug":
            colors.append("#e63946")   # 红
        else:
            colors.append("#cccccc")   # 灰
    return colors


def get_edge_colors(G):
    colors = []
    for _, _, data in G.edges(data=True):
        edge_type = data.get("edge_type", "unknown")
        if edge_type == "contains":
            colors.append("#7b2cbf")   # 紫
        elif edge_type == "calls":
            colors.append("#d00000")   # 红
        elif edge_type == "imports":
            colors.append("#005f73")   # 深青
        else:
            colors.append("#666666")
    return colors


def draw_subgraph(
    G,
    title="Code Graph Local View",
    save_path="graph_subgraph.png",
    show_edge_labels=True
):
    plt.figure(figsize=(16, 12))

    # spring_layout 适合这种小规模图
    pos = nx.spring_layout(G, seed=42, k=1.1)

    node_colors = get_node_colors(G)
    edge_colors = get_edge_colors(G)

    labels = nx.get_node_attributes(G, "label")
    edge_labels = nx.get_edge_attributes(G, "edge_type")

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=1400,
        alpha=0.95
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        arrows=True,
        arrowsize=18,
        width=1.8,
        alpha=0.85,
        connectionstyle="arc3,rad=0.08"
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=8
    )

    if show_edge_labels:
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_size=7,
            rotate=False
        )

    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    print(f"[Saved] {save_path}")
    plt.show()


def find_node_by_name(graph, name, node_type=None):
    """
    按名字查找节点。可指定 node_type。
    """
    matches = []
    for node_id, node in graph.nodes.items():
        if node.name == name:
            if node_type is None or node.type == node_type:
                matches.append((node_id, node))
    return matches


def find_file_node_by_path_keyword(graph, keyword):
    """
    用 file_path 关键词找 file 节点。
    """
    matches = []
    for node_id, node in graph.nodes.items():
        if node.type == "file" and keyword in node.file_path:
            matches.append((node_id, node))
    return matches


def choose_start_node(graph, mode="random_function", query=None):
    """
    mode:
        - random_function
        - function_name
        - file_keyword
    """
    if mode == "random_function":
        function_nodes = [
            (node_id, node)
            for node_id, node in graph.nodes.items()
            if node.type == "function"
        ]
        if not function_nodes:
            raise ValueError("No function nodes found.")
        return random.choice(function_nodes)

    if mode == "function_name":
        matches = find_node_by_name(graph, query, node_type="function")
        if not matches:
            raise ValueError(f"No function named '{query}' found.")
        print(f"[Info] Found {len(matches)} function matches for '{query}', using the first one.")
        return matches[0]

    if mode == "file_keyword":
        matches = find_file_node_by_path_keyword(graph, query)
        if not matches:
            raise ValueError(f"No file containing '{query}' found.")
        print(f"[Info] Found {len(matches)} file matches for '{query}', using the first one.")
        return matches[0]

    raise ValueError(f"Unknown mode: {mode}")


def print_start_node_info(node_id, node):
    print("\n=== Start Node ===")
    print(f"ID       : {node_id}")
    print(f"Name     : {node.name}")
    print(f"Type     : {node.type}")
    print(f"File Path: {node.file_path}")
    if node.doc:
        print(f"Doc      : {node.doc[:120]}{'...' if len(node.doc) > 120 else ''}")


def main():
    repo_path = "data_storage/repos/astropy"

    builder = GraphBuilder()
    graph = builder.build_from_repo(repo_path)

    print(graph)
    print(f"Total nodes: {len(graph.nodes)}")
    print(f"Total edges: {len(graph.edges)}")

    # =========================
    # 这里改你的可视化配置
    # =========================

    # 选择起点方式：
    # "random_function"
    # "function_name"
    # "file_keyword"
    start_mode = "random_function"

    # 当 start_mode = "function_name" 时，query 例如 "run"
    # 当 start_mode = "file_keyword" 时，query 例如 "coordinates" 或 "setup.py"
    query = None

    # 选择边类型：
    # None -> 全部
    # {"calls"}
    # {"contains"}
    # {"imports"}
    # {"calls", "contains"}
    selected_edge_types = {"calls", "contains", "imports"}

    max_depth = 2
    max_nodes = 50

    save_path = "graph_local_view.png"

    # =========================

    start_node_id, start_node = choose_start_node(graph, mode=start_mode, query=query)
    print_start_node_info(start_node_id, start_node)

    sub_node_ids = extract_subgraph_bfs(
        graph,
        start_node_id=start_node_id,
        max_depth=max_depth,
        max_nodes=max_nodes,
        edge_types=selected_edge_types
    )

    G = build_networkx_subgraph(
        graph,
        node_ids=sub_node_ids,
        edge_types=selected_edge_types
    )

    edge_desc = "all" if selected_edge_types is None else ",".join(sorted(selected_edge_types))
    title = (
        f"Local Code Graph\n"
        f"start={start_node.name} [{start_node.type}] | "
        f"depth={max_depth} | max_nodes={max_nodes} | edges={edge_desc}"
    )

    draw_subgraph(
        G,
        title=title,
        save_path=save_path,
        show_edge_labels=True
    )


if __name__ == "__main__":
    main()