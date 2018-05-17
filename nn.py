import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy
from random import randint

input_size = 6       # The image size = 28 x 28 = 784
hidden_size = 3      # The number of nodes at the hidden layer
num_classes = 1       # The number of output classes. In this case, from 0 to 9
num_epochs = 10         # The number of times entire dataset is trained
batch_size = 1       # The size of input data took for one iteration
learning_rate = 0.01 # The speed of convergence

def toBin(x):
  l=list(bin(int(x))[2:])
  while len(l)<6:
    l.insert(0, 0)
  return numpy.array([numpy.int64(int(i)) for i in l])

labels = torch.from_numpy(numpy.array([randint(0,63) for _ in range(5000)])).type(torch.FloatTensor)
train = torch.from_numpy(numpy.array([toBin(i) for i in labels])).type(torch.FloatTensor)

def data_batcher(X, Y, batch_size=25):
    '''
    helper function to batch data, batch_size is the size of 
    the batch.
    In:
        X: a matrix of size (num_sample, CONTEXT_SIZE*2)
        Y: an array of size (num_sample)
        batches: how many data points are in a batche, default:50
    Out:
        a batch of X and Y
    '''
    indices = numpy.arange(len(X))
    # np.random.shuffle(indices)
    count = 0
    while count < len(X):
        draw = indices[count:min(count+batch_size, len(X))]
        yield X[draw, : ], Y[draw]
        count += batch_size


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()                    # Inherited from the parent class nn.Module
        self.fc1 = nn.Linear(input_size, hidden_size)  # 1st Full-Connected Layer: 784 (input data) -> 500 (hidden node)
        self.relu = nn.ReLU()                          # Non-Linear ReLU Layer: max(0,x)
        self.fc2 = nn.Linear(hidden_size, num_classes) # 2nd Full-Connected Layer: 500 (hidden node) -> 10 (output class)
    
    def forward(self, x):                              
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

net = Net(input_size, hidden_size, num_classes)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    print epoch
    for x, y in data_batcher(train, labels, 25):   # Load a batch of images with its (index, data, class)
        x = Variable(torch.stack(x))
        y = Variable(y)
        optimizer.zero_grad()
        out = net.forward(x)
        #print out, y
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()


def f(x):
    train = torch.from_numpy(toBin(x)).type(torch.FloatTensor)
    out = net.forward(Variable(torch.stack([train])))                             # Forward pass: compute the output class given a image
    #print 'NEW', x, train.tolist(), out
    return out

ls = 0
n =0
for j in range(64):
  n = f(j)
  print j, n.data[0]
  ls+= criterion(n, Variable(torch.FloatTensor([j])))

print 1-ls.data[0]/64


