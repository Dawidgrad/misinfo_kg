class KnowledgeGraph:

    def __init__(self) -> None:
        self.nodes = []
        self.edges = []

    def add_relation(self, source, target, edge):

        return None

    def export_csv(self):

        return None

class Node:

    def __init__(self, name, id) -> None:
        self.name = name
        self.id = id

class Edge:

    def __init__(self, source, target, id) -> None:
        self.source = source
        self.target = target
        self.id = id
        self.type = 'Directed'
        self.weight = 1.0
