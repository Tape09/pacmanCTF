class Node:
    def __init__(self,n):
        self.n = n

def changeNode(node):
    node.n +=1

node1 = Node(3)
node2 = Node(1000)
nodes = [node1,node2]
# print (node1.n)
# changeNode(node1)
# print node1.n
for node in nodes:
    changeNode(node)
for node in nodes:
    print node.n
