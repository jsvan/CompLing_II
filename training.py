# where we train for days
import nn as model
import pickle

def trainM1(trainFile,devFile,devTestFile):
	with open(trainFile,"rb") as f:
		train,dont = pickle.load(f)
	with open(devFile,"rb") as f:
		dev,dont = pickle.load(f)
	with open(devTestFile,"rb") as f:
		dev += pickle.load(f)[0]
	net = 