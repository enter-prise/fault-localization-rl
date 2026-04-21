# graph/schema.py

class Node:
    def __init__(self, node_id, name, node_type, code="", doc="", file_path=""):
        self.id = node_id
        self.name = name
        self.type = node_type
        self.code = code
        self.doc = doc
        self.file_path = file_path

    def __repr__(self):
        return (
            f"Node(id={self.id}, name={self.name}, "
            f"type={self.type}, file_path={self.file_path})"
        )


class Edge:
    def __init__(self, src, dst, edge_type):
        self.src = src
        self.dst = dst
        self.type = edge_type

    def __repr__(self):
        return f"Edge({self.src} -> {self.dst}, type={self.type})"


class CodeGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.adj = {}
        self.edge_index = {}

        # 新增：用于去重
        self.edge_set = set()

    def add_node(self, node):
        self.nodes[node.id] = node
        if node.id not in self.adj:
            self.adj[node.id] = []
        if node.id not in self.edge_index:
            self.edge_index[node.id] = []

    def add_edge(self, edge):
        edge_key = (edge.src, edge.dst, edge.type)

        # 去重：如果已经有了，就不再添加
        if edge_key in self.edge_set:
            return

        self.edge_set.add(edge_key)
        self.edges.append(edge)

        if edge.src not in self.adj:
            self.adj[edge.src] = []
        self.adj[edge.src].append(edge.dst)

        if edge.src not in self.edge_index:
            self.edge_index[edge.src] = []
        self.edge_index[edge.src].append(edge)

    def get_neighbors(self, node_id, edge_type=None):
        if edge_type is None:
            return self.adj.get(node_id, [])

        return [
            edge.dst
            for edge in self.edge_index.get(node_id, [])
            if edge.type == edge_type
        ]

    def get_out_edges(self, node_id):
        return self.edge_index.get(node_id, [])

    def __repr__(self):
        return f"CodeGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"