from nltk import word_tokenize,pos_tag,download
from autocorrect import spell
import user_week_buckets as uwb
from data_utils import *

EXCLUDE = {"Anger","BPD","EatingDisorders","MMFB","StopSelfHarm","SuicideWatch","addiction","alcoholism",\
			"depression","feelgood","getting_over_it","hardshipmates","mentalhealth","psychoticreddit",\
			"ptsd","rapecounseling","schizophrenia","socialanxiety","survivorsofabuse","traumatoolbox"}
TOTAL_LIWC = 18
TRAINFS = ['../umd_reddit_suicidewatch_dataset/reddit_posts/controls/split_80-10-10/TRAIN.txt', 'umd_reddit_suicidewatch_dataset/reddit_posts/sw_users/split_80-10-10/TRAIN.txt']
TESTFS = ['../umd_reddit_suicidewatch_dataset/reddit_posts/controls/split_80-10-10/TEST.txt','umd_reddit_suicidewatch_dataset/reddit_posts/sw_users/split_80-10-10/TEST.txt']
DEVFS = ['../umd_reddit_suicidewatch_dataset/reddit_posts/controls/split_80-10-10/DEV.txt','umd_reddit_suicidewatch_dataset/reddit_posts/sw_users/split_80-10-10/DEV.txt']
ANNOFS = ['../umd_reddit_suicidewatch_dataset/reddit_annotation/crowd.csv','umd_reddit_suicidewatch_dataset/reddit_annotation/expert.csv']

#[postid, userid, timestamp, subreddit]


def _processDataset(dataFiles,liwcFile,stopFile):
	'''
	:param dataFiles:
	:param liwcFile:
	:param stopFile:
	:return: pickles post data object
	:return: pickles all the text as a list of tokenized words
	:return: pickles suicide times dict
	'''
	print("A")
	with open(liwcFile,"rb") as lfile:
		liwc = pickle.load(lfile)
	msDict = dict()	
	dataFilenames = list()
	for dataFilePtrn in dataFiles:
		dataFilenames += glob(dataFilePtrn)
	for dataFile in dataFilenames:
		print(dataFile)
		with open(dataFile,"rU",errors="surrogateescape") as data:
			allText = list()
			allPosts = list()
			suicideTimes = dict()
			for post in data: #post string, a line from file
				# print('*', end='', flush=True)
				post = post.strip().split("\t")
				if len(post) > 4: #post a list of strings (post info)
					titleLast = post[4][-1:]
					if titleLast.isalnum(): #i.e. not a punctuation mark:
						post[4] += "."
					post = post[:4] + [" ".join(post[4:])]
					subreddit = post[3]
					if subreddit in EXCLUDE:
						allText += [spellcheck(wrd.lower(),False,msDict) for wrd in word_tokenize(post[4])]
						allText.append("$|$")
						allPosts.append("IGNORE")
						if subreddit == "SuicideWatch":
							suicideTimes[post[1]] = suicideTimes.get(post[1],list()) + [int(post[2])]
					else:
						features = [0]*31
						features[0] = post[1]
						features[-2] = int(post[2])
						features[1] = subreddit
						features = processPostText(post[4],allText,msDict,liwc,features)
						weekend, daytime = timeToDate(int(post[2]))
						features[-4] = weekend
						features[-3] = daytime
						allPosts.append(features)
	print('Pickling')
	with open("allText.p", "wb") as f:
		pickle.dump(allText, f)
	with open("allPosts.p", "wb") as f:
		pickle.dump(allPosts, f)
	with open("suicideTimes.p", "wb") as f:
		pickle.dump(suicideTimes, f)
	return allPosts, allText, suicideTimes

def _allText2TopicModel(allText, stopFile):
	'''
	:param allText: list of each post represented in topic space as a vector (i.e. list of floats)
	:param stopFile: list of each post represented by extracted features (i.e. list of mostly ints, with a gap for topic vector)
	:return docTopicVecs: list of each post represented in topic space as a vector (i.e. list of floats)
	:return ntopics: is an int for num of topics in topic space
	'''
	print('B')
	docTopicVecs,ntopics = collocateAndLDA(allText,stopFile)
	print("Pickling")
	with open("docTopicVecs.p","wb") as f:
		pickle.dump(docTopicVecs,f)
	return docTopicVecs, ntopics


def _addTopicVectorDataAndGroupByUser(docTopicVecs,ntopics,allPosts):
	'''
	:param docTopicVecs: list of each post represented in topic space as a vector (i.e. list of floats)
	:param ntopics: int repping number of topics in topic space
	:param allPosts: list of each post represented by extracted features (i.e. list of mostly ints, with a gap for topic vector)
	:param allocationDict: dict of userid:(dataPartition,prelimLabel) where dataPartition is an int for train,test,dev,devtest and prelimLabel is 0,1,-1
	:return userDict: dict of userid:[posts as vector of extracted features including topic space vector]
	:return mentalHealthVec: vector representation of mental health subreddits (averaged) in topic space
	:return subredditVecDict: dict of subreddit:vector where vector is a representation (averaged) of each subreddit in topic space
	'''
	print("C")
	longVeclen = ntopics + TOTAL_LIWC

	mentalHealthVec = [0] * ntopics #represents avg topic space vecotr in mental health
	totMH = 0
	subredditVecDict = dict() #Flat list of numbers {subreddit_j -> [funcwrdcts and liwc ... topicSpaceVec]}
	userDict = dict() #userid:[posts as vector of extracted features including topic space vector]
	idx = 0
	for post in allPosts:
		if post == "IGNORE":
			mentalHealthVec = [mentalHealthVec[i]+docTopicVecs[i] for i in range(ntopics)]
			totMH += 1
		else:
			subreddit = post[1]
			subredditVec,count,words = subredditVecDict.get(subreddit,([0]*longVeclen,0,0))
			longVec = post[8:8+TOTAL_LIWC]+docTopicVecs[idx]
			subredditVecDict[subreddit] = ([subredditVec[i]+longVec[i] for i in range(longVeclen)],count+1,words+post[2])
			post[-5] = docTopicVecs[idx]
			userDict[post[0]] = userDict.get(post[0],[post])	 
		idx += 1

	mentalHealthVec = [mentalHealthVec[i]/totMH for i in range(ntopics)]

	for (subreddit, (vec,n,w)) in subredditVecDict.items():
		subredditVecDict[subreddit] = [vec[i]/w for i in range(TOTAL_LIWC)]+[vec[i]/n for i in range(TOTAL_LIWC,longVeclen)]

	print("Pickling")
	with open("mentalHealthVec.p","wb") as tp:
		pickle.dump(mentalHealthVec,tp)
	with open("subredditVecs.p","wb") as tp:
		pickle.dump(subredditVecDict,tp)
	with open("userDict.p","wb") as tp:
		pickle.dump(userDict)
	return userDict,mentalHealthVec,subredditVec



def _interpretFeatsAndAllocate(userDict,mentalHealthVec,subredditVecDict,suicideTimes,ntopics,allocationDict):
	'''
	:param userDict: dict of userid:[posts as vector of extracted features including topic space vector]
	:param mentalHealthVec: vector representation of mental health subreddits (averaged) in topic space
	:param subredditVecDict: dict of subreddit:vector where vector is a representation (averaged) of each subreddit in topic space
	:param suicideTimes: dict of userid:[timestamps of posts to r/SuicideWatch as ints]
	:param ntopics: int repping number of topics in topic space
	:param allocationDict: dict of userid:(dataPartition,prelimLabel) where dataPartition is an int for train,test,dev,devtest and prelimLabel is 0,1,-1
	:return trainPosts: ([List of trueLabelled buckets],[list of unlabelled buckets])
	:return testPosts: [List of trueLabelled buckets]
	:return devPosts:  ([List of trueLabelled buckets],[list of unlabelled buckets])
	:return devTestPosts: ([List of trueLabelled buckets],[list of unlabelled buckets])
	'''	
	print('D')
	trainPosts = (list(),list())
	testPosts = list()
	devPosts = (list(),list())
	devTestPosts = (list(),list())		
	for user,postList in userDict.items():
		val,lab = allocationDict.get(user,(0,-1))
		for post in postList:
			post[-1] = lab
		bucketList = uwb.interpret_post_features_by_user(postList, suicideTimes, subredditVecDict, mentalHealthVec,ntopics)	
		if val == 1:
			testPosts += bucketList
		else:
			for bucket in bucketList:
				lab = 0 == bucket[0][-1]
				if val == 0:
					trainPosts[lab].append(bucket)
				elif val == 2:
					devPosts[lab].append(bucket)
				else:
					devTestPosts[lab].append(bucket)
	print("Pickling")
	with open("trainingData.p","wb") as f:
		pickle.dump(trainPosts,f)
	with open("testData.p","wb") as f:
		pickle.dump(testPosts,f) 
	with open("devData.p","wb") as f:
		pickle.dump(devPosts,f) 
	with open("devTestData.p","wb") as f:
		pickle.dump(devTestPosts,f)
	return trainPosts,testPosts,devPosts,devTestPosts

#[userid,subreddit,totw,totmissp,tot1sg,totpron,totpres,totvrb,[funcwrdcts and liwc],[topicSpaceVec],wkday,hr,timestamp,label]
def _processPostText(post, docFile, msdict, liwcDict, featureList):
	wrdList = [spellcheck(wrd.lower(),featureList,msdict) for wrd in word_tokenize(post)]
	docFile += wrdList
	docFile.append("$|$")
	tags = pos_tag(wrdList)
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
	return featureList

def spellcheck(wrd,lst,msdict):
	if (len(wrd) < 20) and wrd.isalpha():
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






'''post from TEXT FILE
  RAW POST

  [post_id]
  [user_id]
  [timestamp]
  [subreddit]
  [post_title]
  [post_body]'''


def prepare():
	# processDataset([argv[1]], "./liwc.p", "engStops")
	# allocationDict = makeAllocationDict(TRAINFS, TESTFS, DEVFS, ANNOFS)
	# with open("allocator.p","wb") as f:
	# 	pickle.dump(allocationDict,f)
	with open("allocator.p",'rb') as f:
		allocator = pickle.load(f)
	a,b,c = stitchTogether(argv[1],argv[2],argv[3])
	_allText2TopicModel(allocator, a, b, c, "engStops")
	_nextStuff()

	#Lets have every method pickle the results, but pass the data structures in memory
	#Each step checks if stuff in memory, if not, creates data from previous step by opening files from pickle
	#
	return #list of Two Week Representations


