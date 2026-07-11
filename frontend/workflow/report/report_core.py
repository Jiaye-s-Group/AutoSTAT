class ReportNode:
    """A report tree node."""

    def __init__(self, node_type, text, level=0):
        self.type = node_type
        self.text = text
        self.level = level
        self.children = []

    def to_dict(self):
        return {
            "type": self.type,
            "text": self.text,
            "level": self.level,
            "children": [c.to_dict() for c in self.children]
        }

class Reportcore:
    """Simple ordered report tree used by legacy export helpers."""

    def __init__(self):
        self.root = ReportNode("root", "", level=-1)
        self.current_stack = [self.root]

    def add_heading(self, text, level=0):
        """Add a heading under the nearest valid parent."""
        new_node = ReportNode("heading", text, level)

        # Walk back to the nearest parent level.
        while self.current_stack and self.current_stack[-1].level >= level:
            self.current_stack.pop()

        parent = self.current_stack[-1]
        parent.children.append(new_node)
        self.current_stack.append(new_node)

    def add_paragraph(self, text):
        """Add a paragraph under the current heading."""
        parent = self.current_stack[-1]

        parent.children.append(ReportNode("paragraph", text, level=parent.level + 1))

    def to_dict(self):
        return self.root.to_dict()
