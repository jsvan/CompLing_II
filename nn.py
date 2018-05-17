import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy
from random import choice, random

input_size = 1       #
hidden_size = 5      # The number of nodes at the hidden layer
num_classes = 2       # The number of output classes. -1, 1
num_epochs = 20         # The number of times entire dataset is trained
batch_size = 1       # The size of input data took for one iteration
learning_rate = 0.01 # The speed of convergence
num_data=18000
num_data_d2=num_data/2


def toBin(x):
  l=list(bin(int(x))[2:])
  while len(l)<6:
    l.insert(0, 0)
  return numpy.array([numpy.int64(int(i)) for i in l])


labels = torch.from_numpy(numpy.array([[0 if j<num_data_d2 else 1] for j in range(num_data)])).type(torch.LongTensor)
train = torch.from_numpy(numpy.array([[float(j)/float(num_data) for _ in range(input_size)] for j in range(num_data)])).type(torch.FloatTensor)

def tupleToTensor(a):
    tens = torch.zeros(1, len(a))
    for i in a:
        tens[0][i] = i
    return tens

def categoryToTensor(a):
    a = numpy.int64(int(a))
    tens = torch.zeros(1, 1)
    tens[0][0] = a   # 1 prime 0 not
  #  tens[0][1] = 1-a # 0 prime 1 not
    return tens


def data_batcher(X, Y, batch_size=25):
    indices = numpy.arange(num_data)
    count = 0
    while count < len(X):
        draw = indices[count:min(count+batch_size, len(X))]
        yield X[draw, :], Y[draw]
        count += batch_size


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()                          # Non-Linear ReLU Layer: max(0,x)
        self.i2o = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim =1)


    def forward(self, input):
        hidden = self.i2h(input)
        out = self.i2o(self.relu(hidden))
        return self.softmax(out)

net = RNN(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


def go():
    net.zero_grad()
    '''
    for epoch in range(num_epochs):
        totLoss=0.0
        print(epoch)
        for j in range(len(labels)):
            x, y = train[j], labels[j]
       # for x, y in data_batcher(train, labels[j]):

            X=Variable(tupleToTensor(x))
            out = net.forward(X)
            Y = Variable(categoryToTensor(y).type(torch.LongTensor))
            out = out
            optimizer.zero_grad()
            q, w = out.max(1)
            printDots(Y.data[0].numpy()[0]-w.data[0])
            loss = criterion(out, Y.squeeze()) #Needs a 1d list of correct index. Float -> Long
            totLoss+=loss
            loss.backward(retain_graph=True)
            optimizer.step()
        print totLoss.data[0]/len(labels)
    '''
    for epoch in range(num_epochs):
        print(epoch)
        totLoss=0.0
        for x, y in data_batcher(train, labels, 25):   # Load a batch of images with its (index, data, class)
            x = Variable(x)
            y = Variable(y)
            optimizer.zero_grad()
            out = net.forward(x)

            loss = criterion(out, y.squeeze())
            totLoss+=loss
            loss.backward()
            optimizer.step()

        print(totLoss.data.numpy()/len(labels))



'''

'''
go()

ls = 0.0
n =0
tf='F'
med=0.0
avg=0.0
print("\n\n\n***TESTING***\n\n\n")

ltest = torch.from_numpy(numpy.array([[0 if j<50 else 1] for j in range(100)])).type(torch.LongTensor)
ttest = torch.from_numpy(numpy.array([[float(j)/float(100) for _ in range(input_size)] for j in range(100)])).type(torch.FloatTensor)

for x, y in data_batcher(ttest, ltest, 25):

    print('VAR X')
    X=Variable(x)
    print(X)
    print('VAR Y')
    print(Variable(y).squeeze())
    out = net.forward(X)
    loss =criterion(out, Variable(y).squeeze())
    ls+=loss
    print("OUT")
    print(out)
print(ls/100.0)
