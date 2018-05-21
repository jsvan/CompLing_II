import preprocessing
import nn
from data_utils import pearsonsR

if __name__ == '__main__':
	trainPosts, testPosts, devPosts, devTestPosts = preprocessing.prepare()
	pearsonsR(trainPosts[0])
	nn = nn.simple_feed_forward()