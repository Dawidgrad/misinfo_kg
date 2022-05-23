import pandas as pd

class KnowledgeGraph:

    def __init__(self) -> None:
        self.nodes = []
        self.edges = []

    def add_relation(self, subject, verb, object):
        source = Node(subject, len(self.nodes))

        # Check if source node already exists, if not add it to the list
        source = self.validate_node(source)
        if source.id == len(self.nodes):
            self.nodes.append(source)

        target = Node(object, len(self.nodes))

        # Do the same for target node
        target = self.validate_node(target)
        if target.id == len(self.nodes):
            self.nodes.append(target)

        relationship = Edge(source.id, target.id, verb, len(self.edges))
        self.edges.append(relationship)

        return None

    def export_csv(self, working_dir):
        nodes_df = pd.DataFrame([{'Id': x.id, 'Label': x.name} for x in self.nodes])
        nodes_df.to_csv(f'{working_dir}/source/output_files/nodes.csv', index=False)
        
        edges_df = pd.DataFrame([{'Source': x.source_id, 'Target': x.target_id, 'Type': x.type, 
            'Id': x.id, 'Label': x.label, 'Weight': x.weight} for x in self.edges])
        edges_df.to_csv(f'{working_dir}/source/output_files/edges.csv', index=False)

        return None
    
    def validate_node(self, new_node):
        result = new_node

        for node in self.nodes:
            if new_node.name == node.name:
                result = node

        return result

class Node:

    def __init__(self, name, id) -> None:
        self.name = name
        self.id = id

class Edge:

    def __init__(self, source_id, target_id, label, id) -> None:
        self.source_id = source_id
        self.target_id = target_id
        self.label = label
        self.id = id
        self.type = 'Directed'
        # TODO Adjust the weight based on the amount of times same combination appears?
        self.weight = 1.0
