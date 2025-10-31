class ReportNode:
    """Document node: can be a heading or a paragraph"""
    def __init__(self, node_type, text, level=0):
        self.type = node_type   # "heading" or "paragraph"
        self.text = text
        self.level = level
        self.children = []  # Child nodes (for hierarchical structure)

    def to_dict(self):
        return {
            "type": self.type,
            "text": self.text,
            "level": self.level,
            "children": [c.to_dict() for c in self.children]
        }

# Currently only suitable for sequential addition
class Reportcore:
    def __init__(self):
        self.root = ReportNode("root", "", level=-1)  # Virtual root node
        self.current_stack = [self.root]  # Use stack to manage current level

    def add_heading(self, text, level=0):  # Starting from 0
        """
        Add a heading, automatically mounts to the appropriate parent node based on level
        """
        new_node = ReportNode("heading", text, level)

        # Backtrack to the appropriate parent node
        while self.current_stack and self.current_stack[-1].level >= level:
            self.current_stack.pop()

        parent = self.current_stack[-1]
        parent.children.append(new_node)
        self.current_stack.append(new_node)

    def add_paragraph(self, text):
        """
        Add a paragraph, attached under the last current heading
        """
        parent = self.current_stack[-1]

        parent.children.append(ReportNode("paragraph", text, level=parent.level + 1))

    def to_dict(self):
        return self.root.to_dict()