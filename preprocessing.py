from nltk import word_tokenize
from nltk.tag import stanford as stanf
from autocorrect import spell
import pickle
import datetime
from gensim import corpora, models, similarities
from scipy.stats.stats import pearsonr as pr
from glob import glob
from nltk.collocations import *
from nltk.util import ngrams
from nltk.metrics import BigramAssocMeasures
from string import punctuation as punct

TWO_WEEKS = 1209600
EXCLUDE = {"Anger","BPD","EatingDisorders","MMFB","StopSelfHarm","SuicideWatch","addiction","alcoholism",\
			"depression","feelgood","getting_over_it","hardshipmates","mentalhealth","psychoticreddit",\
			"ptsd","rapecounseling","schizophrenia","socialanxiety","survivorsofabuse","traumatoolbox"}
TOTAL_LIWC = 18
TRAINFS = []
TESTFS = []
DEVFS = []
ANNOFS = []

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
#[postid, userid, timestamp, subreddit]
def processDataset(dataFiles,liwcFile,stopFile):
	tagger = stanf.StanfordPOSTagger("/Users/owner/stanford-postagger-full-2018-02-27/models/english-caseless-left3words-distsim.tagger")
	with open(liwcFile,"rb") as lfile:
		liwc = pickle.load(lfile)
	allocationDict = makeAllocationDict(TRAINFS, TESTFS, DEVFS, ANNOFS)
	msDict = dict()	
	allText = list()
	allPosts = list()
	dataFilenames = list()
	suicideTimes = dict()
	for dataFilePtrn in dataFiles:
		dataFilenames += glob(dataFilePtrn)
	for dataFile in dataFilenames:
		with open(dataFile,"rU",errors="surrogateescape") as data:
			for post in data:
				post = post.strip()
				if post != "":
					post = post.split("\t")
					titleLast = post[4][-1:]
					if titleLast.isalnum(): #i.e. not a punctuation mark:
						post[4] += "."
					post[4] = " ".join(post[4:])
					subreddit = post[3]
					if subreddit in EXCLUDE:
						allText += (spellcheck(wrd.lower(),False,msdict) for wrd in word_tokenize(post[5]))
						allText.append("$|$")
						allPosts.append("IGNORE")
						if subreddit == "SuicideWatch":
							suicideTimes[post[1]] = suicideTimes.get(post[1],list()).append(int(post[2]))
					else:
						features = [0]*30
						features[0] = post[1]
						features[-2] = int(post[2])
						features[1] = subreddit
						features = processPostText(post[4],allText,tagger,msDict,liwc,features)
						allPosts.append(features)
	docTopicVecs = collocateAndLDA(allText,stopFile)
	ntopics = len(docTopicVecs[0])
	longVeclen = ntopics + TOTAL_LIWC
	trainPosts = list()
	testPosts = list()
	devPosts = list()
	devTestPosts = list()
	mentalHealthVec = [0] * ntopics #represents avg topic space vecotr in mental health
	totMH = 0
	subredditVecDict = dict() #Flat list of numbers {subreddit_j -> [funcwrdcts and liwc ... topicSpaceVec]}
	idx = 0
	for post in allPosts:
		if post == "IGNORE":
			mentalHealthVec = [mentalHealthVec[i]+docTopicVecs[i] for i in range(ntopics)]
			totMH += 1
		else:
			subreddit = post[1]
			subredditVec,count,words = subredditVecDict.get(subreddit,([0]*longVeclen,0,0))
			longVec = post[8:8+TOTAL_LIWC]+docTopicVecs[i]
			subredditVecDict[subreddit] = ([subredditVec[i]+longVec[i] for i in range(longVeclen)],count+1,words+post[2])
			post[-5] = docTopicVecs[idx]
			val,lab = allocationDict.get(post[0],(0,-1))
			post[-1] = lab
			if val == 0:
				trainPosts.append(post)
			elif val == 1:
				testPosts.append(post)
			elif val == 2:
				devPosts.append(post)
			else:
				devTestPosts.append(post)		 
		idx += 1
	mentalHealthVec = [mentalHealthVec[i]/totMH for i in range(ntopics)]
	for (subreddit, (vec,n,w)) in subredditVecDict.items():
		subredditVecDict[subreddit] = [vec[i]/w for i in range(TOTAL_LIWC)]+[vec[i]/n for i in range(TOTAL_LIWC,longVeclen)]
	with open("trainPosts.p","wb") as tp:
		pickle.dump(trainPosts,tp)
	with open("testPosts.p","wb") as tp:
		pickle.dump(testPosts,tp)
	with open("devPosts.p","wb") as tp:
		pickle.dump(devPosts,tp)
	with open("devTestPosts.p","wb") as tp:
		pickle.dump(devTestPosts,tp)
	with open("mentalHealthVec.p","wb") as tp:
		pickle.dump(mentalHealthVec,tp)
	with open("subredditVecs.p","wb") as tp:
		pickle.dump(subredditVecDict,tp)
	with open("suicideTimes.p","wb") as tp:
		pickle.dump(suicideTimes,tp)

#[userid,subreddit,totw,totmissp,tot1sg,totpron,totpres,totvrb,[funcwrdcts and liwc],[topicSpaceVec],wkday,hr,timestamp,label]
def processPostText(post, docFile, tagger, msdict, liwcDict, featureList):
	wrdList = (spellcheck(wrd.lower(),featureList,msdict) for wrd in word_tokenize(post))
	docFile += wrdList
	docFile.append("$|$")
	tags = tagger.tag(wrdList)
	for wrd, tag in tags:
		if tag[0:1] == "V":
			featureList[7] += 1
			if tag in {"VBG","VBP","VBZ"}:
				featureList[6] += 1
		elif tag[0:3] == "PRP":
			featureList[5] += 1
			if wrd in {"me","my","I","myself","mine"}:
				featureList[4] += 1
		elif wrd in liwcDict:
			themes=liwcDict[wrd]
			for theme in themes:
				featureList[8+theme] += 1
	weekend,daytime = timeToDate(post[2])
	featureList[-4] = weekend
	featureList[-3] = daytime
	return featureList

def spellcheck(wrd,lst,msdict):
	if wrd.isalpha():
		if wrd in msdict:
			new = msdict[wrd]
		else:
			new = spell(wrd).lower()
			msdict[wrd] = new
		if lst:
			lst[2] += 1
			if new != wrd:
				lst[3] += 1
		return new
	else:
		return wrd

def appendTimeToDate(time):
	weekend = 0
	s= datetime.datetime.fromtimestamp(time)
	dayOfTheWeek=s.strftime("%a")
	if dayOfTheWeek=='Sat' or dayOfTheWeek=='Sun':
		weekend = 1
	hour24=s.strftime("%H")
	bucket = floor(hour24/6)
	return (weekend,bucket)

def collocateAndLDA(allWords):
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

def applyFilters(bigrammer,filterList):
	for f in filterList:
		f(bigrammer)
	return bigrammer

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

def pearsonsR(masterlist):
	# input masterlist[featurelist, featurelist, featurelist, featurelist ...]
	# where featurelist has the label at last spot
	# want pr(x,y) to output (1.0, 0.0) or (-1.0, 0.0). (0, 1) is very uncorrelated.
	y = [instance[-1] for instance in masterList]
	vals = [pr(x,y) for x in [[instance[j] for instance in masterList] for j in range(featlen)]]

def toLdaModel(docLists,num_topics):
	dictionary = corpora.Dictionary(docLists)
	corpus = [dictionary.doc2bow(docList) for docList in docLists]
	model = models.LdaModel(corpus, dictionary, num_topics)
	return [[t[1] for t in sorted(model.get_document_topics(doc),key=lambda tup: tup[0])] for doc in corpus]
	
