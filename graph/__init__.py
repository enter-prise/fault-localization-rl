def __init__(self):
    self.node_id = 0

    # 名称到节点 id 的映射
    self.function_name_to_ids: Dict[str, List[str]] = {}
    self.class_name_to_ids: Dict[str, List[str]] = {}
    self.file_path_to_id: Dict[str, str] = {}

    # 新增：更细粒度索引
    self.function_id_to_file: Dict[str, str] = {}
    self.function_id_to_class: Dict[str, Optional[str]] = {}
    self.function_key_to_id: Dict[Tuple[str, str, Optional[str]], str] = {}
    # key: (file_path, func_name, parent_class_name)

    # 延迟解析的边
    self.pending_calls: List[Tuple[str, str]] = []      # (caller_func_id, callee_name)
    self.pending_imports: List[Tuple[str, str]] = []    # (file_id, import_name)