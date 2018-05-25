
import pickle
from gensim import corpora, models, similarities
from scipy.stats.stats import pearsonr as pr
import datetime
import math

'''
A file of self contained methods, used in preprocessing
'''



def pearsonsR(masterList):
	# input masterlist[featurelist, featurelist, featurelist, featurelist ...]
	# where featurelist has the label at last spot
	# want pr(x,y) to output (1.0, 0.0) or (-1.0, 0.0). (0, 1) is very uncorrelated.
	featlen = len(masterList[0]) - 1
	y = [instance[-1] for instance in masterList]
	countNot1 = 0
	count1 = 0
	for yval in y:
		if yval != -1:
			countNot1+=1
		else:
			count1+=1
	print("Not -1: %d" %countNot1)
	print("-1: %d" %count1)
	exes = [[instance[j] for instance in masterList] for j in range(featlen)]
	idx = 0
	for x in exes:
		firstVal = x[0]
		out = True
		for xval in x:
			out = out and (xval == firstVal)
		if out:
			print('problem index ', idx)
		idx += 1
	
	vals = [pr(x,y) for x in exes]
	return vals

def makeCollocated(corp,stops):
	newCorp = list()
	curDoc = list()
	for word in corp:
		if type(word) != str:
			print(word, ' is not a string (data_utils.makeCollocated)')
		elif word == "$|$":
			newCorp.append(curDoc)
			curDoc = list()
		elif word in stops:
			pass
		else:
			curDoc.append(word)
	return newCorp

def applyFilters(bigrammer,filterList):
	for f in filterList:
		f(bigrammer)
	return bigrammer

def collocRecursively(corp,constructor,threshhold,addUnrelated,addBigram,measureFunc,filters=None):
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
		if bgScores.get((bg[0],bg[1]),0) > threshhold:
			addBigram(newCorp,bg)
			idx += 2
			flag = True
		else:
			addUnrelated(newCorp,bg[0])
			idx += 1
	if idx == N-1:
		addUnrelated(newCorp,corp[idx])
	if flag:
		return collocRecursively(newCorp, constructor, threshhold, addUnrelated, addBigram, filters)
	return newCorp

def collocateAndLDA(allWords, stopFile):
	with open(stopFile,"rU") as sf:
		stops = {line.lower().strip() for line in sf.readlines()}
		'''
	constructor = lambda c: BigramCollocationFinder.from_words(c)
	threshhold = 6000
	addUnrelated = lambda c, x: c.append(x)
	addBigram = lambda c, tup: c.append(tup[0]+"_"+tup[1])
	measureFunc = BigramAssocMeasures().likelihood_ratio
	filters = [lambda bg: bg.apply_word_filter(lambda t: (len(t) <2 and not t.isalnum()) or (t in set(punct) | {"$|$"})),\
				 lambda bg: bg.apply_ngram_filter(lambda w1,w2: (w1 in stops) and (w2 in stops))]
	# return toLdaModel(makeCollocated(collocRecursively(\
		# allWords,constructor,threshhold,addUnrelated,addBigram,measureFunc,filters),interpFunc),70)
		'''
	return toLdaModel(makeCollocated(allWords,stops),70)

def timeToDate(time):
	weekend = 0
	s= datetime.datetime.fromtimestamp(time)
	dayOfTheWeek=s.strftime("%a")
	if dayOfTheWeek=='Sat' or dayOfTheWeek=='Sun':
		weekend = 1
	hour24=s.strftime("%H")
	bucket = math.floor(int(hour24)/6)
	return (weekend,bucket)

def makeAllocationDict(trainFiles,testFiles,devFiles,annoteFiles):
	allocationDict = dict() #userid to (int,int): first int for train,test,devtest, or dev, second for label
	for annoteFile in annoteFiles:
		with open(annoteFile) as f:
			f.readline()
			if "expert" in annoteFile:
				allocationDict.update({line[0]:(1,int(line[1])) for line in [l.strip().split(",") for l in f]})
			else:
				allUsrs = f.readlines()
				shuffle(allUsrs)
				t = int(len(allUsrs) * 0.8)
				allocationDict.update({line[0]:(0,int(line[1]))} for line in [l.strip().split(",") for l in allUsrs[0:t]])
				allocationDict.update({line[0]:(2,int(line[1]))} for line in [l.strip().split(",") for l in allUsrs[t::2]])
				allocationDict.update({line[0]:(3,int(line[1]))} for line in [l.strip().split(",") for l in allUsrs[t+1::2]])
	for trainFile in trainFiles:
		default = - ("controls" in trainFile)
		with open(trainFile) as tfile:
			allocationDict.update({line.strip():allocationDict.get(line.strip(),(0,default)) for line in tfile})
	for testFile in testFiles:
		default = - ("controls" in testFile)
		val = - default # 1 (testSet) if controls, 0 (training) if otherwise 
		with open(testFile) as tfile:	
			allocationDict.update({line.strip():allocationDict.get(line.strip(),(val,default)) for line in tfile})
	for devFile in devFiles:
		default = - ("controls" in devFile)
		with open(devFile) as dfile:
			total = dfile.readlines()
			allocationDict.update({line.strip():(allocationDict.get(line.strip(),(2,default))) for line in total[1::2]})
			allocationDict.update({line.strip():(allocationDict.get(line.strip(),(2,default))) for line in total[0::2]})
	with open("allocationDict.p","wb") as f:
		print("Pickling allocator")
		pickle.dump(allocationDict,f)
	return allocationDict

def toLdaModel(docLists, num_topics):
	dictionary = corpora.dictionary.Dictionary(docLists)
	corpus = [dictionary.doc2bow(docList) for docList in docLists]
	model = models.ldamulticore.LdaMulticore(corpus, num_topics,workers=7)
	with open("ldaModel.p","wb") as f:
		pickle.dump(model,f)
	docVecs = list()
	for doc in corpus:
		docVec = [0] * num_topics
		sparse = model.get_document_topics(doc)
		for top,val in sparse:
			docVec[top] = val
		docVecs.append(docVec)
	return docVecs, num_topics

def stitchTogether(numFs):
	'''
	:param postFs:
	:param textFs:
	:param timeFs:
	:return: unpickles everything and returns union of each
	'''
	allPosts = list()
	allText = list()
	suicideTimes = dict()
	for idx in range(numFs):
		with open("allText_%d.p" %idx, "rb") as f:
			allText += pickle.load(f)
		with open("allPosts_%d.p" %idx, "rb") as f:
			allPosts += pickle.load(f)
		with open("suicideTimes_%d.p" %idx, "rb") as f:
			dc = pickle.load(f)
			for usr,lst in dc.items():
				suicideTimes[usr] = suicideTimes.get(usr,list()) + lst
	return allPosts,allText,suicideTimes








