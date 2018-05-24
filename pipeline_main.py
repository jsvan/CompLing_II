import preprocessing
import nn
import pickle
from data_utils import pearsonsR

if __name__ == '__main__':
	trainPosts, testPosts, devPosts, devTestPosts = preprocessing.prepare()
	pr = pearsonsR(trainPosts[0])
	for item in pr:
		print(item)
	with open('pearsonsR.p', 'wb') as f:
		pickle.dump(pr, file=f)
		pickle.dump

	print('Beginning nn')
	nn = nn.simple_feed_forward()
	nn.train(trainPosts)
	nn.test(testPosts)