
from nltk.collocations import *
#from nltk.util import ngrams
from nltk.metrics import BigramAssocMeasures

from gensim import corpora, models, similarities
from scipy.stats.stats import pearsonr as pr
from string import punctuation as punct
import datetime
import math

def pearsonsR(masterList):
	# input masterlist[featurelist, featurelist, featurelist, featurelist ...]
	# where featurelist has the label at last spot
	# want pr(x,y) to output (1.0, 0.0) or (-1.0, 0.0). (0, 1) is very uncorrelated.
	featlen = len(masterList[0])
	y = [instance[-1] for instance in masterList]
	vals = [pr(x,y) for x in [[instance[j] for instance in masterList] for j in range(featlen)]]
	return vals

def makeCollocated(corp,interpFunc):
	newCorp = list()
	curDoc = list()
	for word in corp:
		if interpFunc(word) == "$|$":
			newCorp.append(curDoc)
			curDoc = list()
		else:
			curDoc.append(word)
	return newCorp

def applyFilters(bigrammer,filterList):
	for f in filterList:
		f(bigrammer)
	return bigrammer


def collocRecursively(corp,interp,constructor,threshhold,addUnrelated,addBigram,measureFunc,filters=None):
	bgFinder = constructor(corp)
	if filters:
		bgFinder = applyFilters(bgFinder,filters)
	bgScores = {bg:score for bg,score in bgFinder.score_ngrams(measureFunc)}
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

def collocateAndLDA(allWords, stopFile):
	with open(stopFile,"rU") as sf:
		stops = {line.strip() for line in sf.readlines()}
	interpFunc = lambda x: x
	constructor = lambda c: BigramCollocationFinder.from_words(c)
	threshhold = 600
	addUnrelated = lambda c, x: c.append(x)
	addBigram = lambda c, tup: c.append(tup[0]+"_"+tup[1])
	measureFunc = BigramAssocMeasures().likelihood_ratio
	filters = [lambda bg: bg.apply_word_filter(lambda t: t in set(punct) | {"$|$"}),\
				 lambda bg: bg.apply_ngram_filter(lambda w1,w2: (w1 in stops) and (w2 in stops))]
	return toLdaModel(makeCollocated(collocRecursively(\
		allWords,interpFunc,constructor,threshhold,addUnrelated,addBigram,measureFunc,filters),interpFunc),70)




def timeToDate(time):
	weekend = 0
	s= datetime.datetime.fromtimestamp(time)
	dayOfTheWeek=s.strftime("%a")
	if dayOfTheWeek=='Sat' or dayOfTheWeek=='Sun':
		weekend = 1
	hour24=s.strftime("%H")
	bucket = math.floor(hour24/6)
	return (weekend,bucket)

def makeAllocationDict(trainFiles,testFiles,devFiles,annoteFiles):
	allocationDict = dict() #userid to (int,int): first int for train,test,devtest, or dev, second for label
	for annoteFile in annoteFiles:
		with open(annoteFile) as f:
			allocationDict.update({line[0]:line[1] for line in [l.strip().split(",") for l in f.readlines()]})
	for trainFile in trainFiles:
		with open(trainFile) as tfile:
			allocationDict.update({line.strip():(0,allocationDict.get(line.strip(),-1)) for line in tfile.readlines()})
	for testFile in testFiles:
		with open(testFile) as tfile:
			allocationDict.update({line.strip():(1,allocationDict.get(line.strip(),-1)) for line in tfile.readlines()})
	for devFile in devFiles:
		with open(devFile) as dfile:
			total = tfile.readlines()
			allocationDict.update({line.strip():(2,allocationDict.get(line.strip(),-1)) for line in total[1::2]})
			allocationDict.update({line.strip():(3,allocationDict.get(line.strip(),-1)) for line in total[0::2]})
	return allocationDict


def toLdaModel(docLists, num_topics):
	dictionary = corpora.dictionary.Dictionary(docLists)
	corpus = [dictionary.doc2bow(docList) for docList in docLists]
	model = models.LdaModel(corpus, dictionary, num_topics)
	return [[t[1] for t in sorted(model.get_document_topics(doc), key=lambda tup: tup[0])] for doc in corpus]

