import numpy as np
from scipy.special import expit as sigmoid

def tanh(x):
    return(1/((np.exp(x)-np.exp(-x))*(np.exp(x)+np.exp(-x))))

def sigma(z):
    return 1/(1+np.exp(-z))

class Obj(object):
    
    def __init__(self, D, I, K): #dunder habit
        self.W = np.random.uniform(low=-0.01,high=0.01,size=(D,I))
        self.R = np.random.uniform(low=-0.01,high=0.01,size=(I))
        self.V = np.random.uniform(low=-0.01,high=0.01,size=(I,K))
        
        t = 5 #any value of t
        test_x = np.random.uniform(size=(D))
        try:
            self.W.T @ test_x
            self.R.T*tanh(t)
            self.V.T*tanh(t)
        except Exception as e:
            print("Model init fail")
        
        
T, D, I, K = 10, 3, 5, 1

model = Obj(D, I, K)
print("W:\n",model.W)
print("R:\n",model.R)
print("V:\n",model.V)

def Loss(y_hat,y):
    print(y_hat,y)
    return (-y*np.log(y_hat)-(1-y)*np.log(1-y_hat))

def forward(self, x, y):
    self.a = [np.zeros(I)]
    self.z = []
    self.y_hat = []
    
    for t,x_t in enumerate(x):#defining t to start at one but keeping a zero-th a makes this tricky
        s = self.W.T@x[t] + self.R.T@self.a[-1]
        self.a.append(tanh(s))
        self.z.append(self.V.T@self.a[-1]) #a[-1] is now the new a
    self.y_hat=map(sigma,self.z)
    print("y_hat")
    for _ in self.y_hat:
        print(_)
    loss_sequence = [ Loss(y_hat[0],y) for y_hat in self.y_hat ]
    print("loss sequence")
    for _ in loss_sequence:
        print(_)
    return sum(loss_sequence)
        

Obj.forward = forward
model = Obj(D, I, K) #have to re-init the instance
model.forward(np.random.uniform(-1, 1, (T, D)), 1)


import networkx as nx
import matplotlib.pyplot as plt

G=nx.Graph()
edgelist = []
"""
for _ in range(D):
    G.add_node(f"x_{_}")
for _ in range(I):
    G.add_node(f"a_{_}")
for _ in range(K):
    G.add_node(f"y_hat_{_}")
 """    
#for _ in range(I):
#    edgelist.append((f"a_{_}",f"a_{_}",model.R[_]))


for column in range(model.W.shape[0]):
    for row in range(model.W.shape[1]):
        edgelist.append((f"x_{column}",f"a_{row}",model.W[column][row]))

for column in range(model.V.shape[0]):
    for row in range(model.V.shape[1]):
        edgelist.append((f"a_{column}",f"y_hat_{row}",model.V[column][row]))

        
G.add_weighted_edges_from(edgelist)        
pos = nx.multipartite_layout(G)
nx.draw_circular(G,with_labels = True,arrowstyle="<|-",)#,edge_color=colors)
plt.show()
