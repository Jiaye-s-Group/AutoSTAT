class ReportNode:
    """文档节点：可以是 heading 或 paragraph"""
    def __init__(self, node_type, text, level=0):
        self.type = node_type   # "heading" 或 "paragraph"
        self.text = text
        self.level = level
        self.children = []  # 子节点（用于分层）

    def to_dict(self):
        return {
            "type": self.type,
            "text": self.text,
            "level": self.level,
            "children": [c.to_dict() for c in self.children]
        }

# 现在只适合于顺序添加
class Reportcore:
    def __init__(self):
        self.root = ReportNode("root", "", level=-1)  # 虚拟根节点
        self.current_stack = [self.root]  # 用栈管理当前层级

    def add_heading(self, text, level=0):# 从0开始
        """
        添加标题，根据 level 自动挂载到合适的父节点
        """
        new_node = ReportNode("heading", text, level)

        # 回溯到合适的父节点
        while self.current_stack and self.current_stack[-1].level >= level:
            self.current_stack.pop()

        parent = self.current_stack[-1]
        parent.children.append(new_node)
        self.current_stack.append(new_node)

    def add_paragraph(self, text):
        """
        添加段落，挂在当前最后一个 heading 下
        """
        parent = self.current_stack[-1]

        parent.children.append(ReportNode("paragraph", text, level=parent.level + 1))

    def to_dict(self):
        return self.root.to_dict()
