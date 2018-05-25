import preprocessing
import nn
import pickle
from data_utils import pearsonsR

if __name__ == '__main__':
	trainPosts, testPosts, devPosts, devTestPosts = preprocessing.prepare()
	countNot1 = 0
	for i in trainPosts[0]:
		if i[-1] != -1:
			countNot1 +=1
	print('Count not -1 is ', countNot1)

	pr = pearsonsR(trainPosts[0])
	for item in pr:
		print(item)
	with open('pearsonsR.p', 'wb') as f:
		pickle.dump(pr, file=f)
		pickle.dump

	print('Beginning nn')
#	nn = nn.simple_feed_forward()
	nn.train(trainPosts[0])
	nn.test(testPosts)