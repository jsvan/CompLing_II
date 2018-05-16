from nltk.collocations import *
from nltk.util import ngrams
from nltk.metrics import BigramAssocMeasures
from string import punctuation as punct

def collocRecursively(corp,interp,constructor,threshhold,addUnrelated,addBigram,filters=None):
	bgFinder = constructor(corp)
	if filters:
		bgFinder = applyFilters(bgFinder,filters)
	bgScores = {bg:score for bg,score in bgFinder.score_ngrams(BigramAssocMeasures().likelihood_ratio)}
	print(sorted(list(bgScores.items()),key=lambda tup: tup[1])[-6:])
	idx = 0
	N = len(corp)
	newCorp = list()
	flag = False
	while idx < N-1:
		bg = (corp[idx],corp[idx+1])
		if bgScores.get((interp(bg[0]),interp(bg[1])),0) > threshhold:
			addBigram(newCorp,bg)
			idx += 2
			flag = True
		else:
			addUnrelated(newCorp,bg[0])
			idx += 1
	if idx == N-1:
		addUnrelated(newCorp,corp[idx])
	if flag:
		return collocRecursively(newCorp, interp, constructor, threshhold, addUnrelated, addBigram, filters)
	return newCorp

def applyFilters(bigrammer,filterList):
	for f in filterList:
		f(bigrammer)
	return bigrammer

def makeCollocated(corp,interpFunc,outfile):
	newCorp = list()
	curDoc = list()
	for word in corp:
		if interpFunc(word) == "$|$":
			newCorp.append(curDoc)
			curDoc = list()
		else:
			curDoc.append(word)

















