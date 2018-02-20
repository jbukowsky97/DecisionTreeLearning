class NeuralNode:
    
    def __init__(self):
        self.children = []
        self.attribute_index = -1
        self.attribute_value = None
        self.decision = None
        self.leaf = False

    def print(self, tabs=""):
        print("%slength children" % tabs, len(self.children))
        print("%sa index" % tabs, self.attribute_index)
        print("%sa value" % tabs, self.attribute_value)
        print("%sa decision" % tabs, self.decision)
        print("%sleaf" % tabs, self.leaf)
        for child in self.children:
            child.print(tabs + "\t")