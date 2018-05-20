import pickle
import numpy as np



def appendpickles(picklefiles):
	features=[]
	labels=[]
	ft=[]
	lt=[]
	for file in picklefiles:
		p= pickle.load(open(file, 'rb'))
		ft, lt = feat_lab(p)
		features+=ft
		labels+=lt
	return features, labels

def feat_lab(lol):
	a = np.array(lol).T
	#labels = a[-1]
	#features = a[:-1]
	return a[:-1], a[-1]