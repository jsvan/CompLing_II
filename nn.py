import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy
from numpy.random import normal

input_size = 24  #
hidden_size_a = 12  # The number of nodes at the hidden layer
hidden_size_b = 6  # The number of nodes at the hidden layer
num_classes = 2  # The number of output classes. -1, 1
num_epochs = 10  # The number of times entire dataset is trained
learning_rate = 0.01  # The speed of convergence



def prepareData(d):
	data = numpy.array(d)
	features = data.T[:-1].T
	labels = data.T[-1]
	print(numpy.shape(data))
	print(numpy.shape(labels))
	print(labels)
	num_data = len(data)

	return  torch.from_numpy(features).type(torch.FloatTensor), torch.from_numpy(labels).type(torch.LongTensor), num_data





def tupleToTensor(a):
	tens = torch.zeros(1, len(a))
	for i in a:
		tens[0][i] = i
	return tens


def categoryToTensor(a):
	a = numpy.int64(int(a))
	tens = torch.zeros(1, 1)
	tens[0][0] = a  # 1 prime 0 not
	#  tens[0][1] = 1-a # 0 prime 1 not
	return tens


def data_batcher(X, Y, num_data, batch_size=25):
	indices = numpy.arange(num_data)
	count = 0
	while count < len(X):
		draw = indices[count:min(count + batch_size, len(X))]
		yield X[draw, :], Y[draw]
		count += batch_size


class simple_feed_forward(nn.Module):
	def __init__(self, input_size, hidden_size_a, hidden_size_b, num_classes):
		super(simple_feed_forward, self).__init__()
		self.hidden_size_a = hidden_size_a
		self.hidden_size_b = hidden_size_b

		self.relu = nn.ReLU()  # Non-Linear ReLU Layer: max(0,x)
		self.i2ha = nn.Linear(input_size, hidden_size_a)
		self.ha2hb = nn.Linear(hidden_size_a, hidden_size_b)
		self.hb2o = nn.Linear(hidden_size_b, num_classes)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, input):
		hidden_a = self.i2ha(input)
		hidden_b = self.ha2hb(self.relu(hidden_a))
		out = self.hb2o(self.relu(hidden_b))
		return self.softmax(out)


def train(data):
	features, labels, num_data  = prepareData(data)
	net.zero_grad()
	for epoch in range(num_epochs):
		print(epoch)
		totLoss = 0.0
		for x, y in data_batcher(features, labels, num_data, 25):  # Load a batch of images with its (index, data, class)
			x = Variable(x)
			y = Variable(y)
			optimizer.zero_grad()
			out = net.forward(x)

			loss = criterion(out, y.squeeze())
			totLoss += loss
			loss.backward()
			optimizer.step()

		print(totLoss.data.numpy() / len(labels))



def test(data):
	features, labels, num_data  = prepareData(data)

	print("\n\n\n***TESTING***\n\n\n")
	ls = 0
	count = 0
	for x, y in data_batcher(features, labels, num_data, 25):
		X = Variable(x)
		out = net.forward(X)
		loss = criterion(out, Variable(y).squeeze())
		ls += loss
		count += 25

	print(ls / num_data)




net = simple_feed_forward(input_size, hidden_size_a, hidden_size_b, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
