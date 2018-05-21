import user_week_buckets as uwb
from data_utils import *
import os.path
from glob import glob
import pickle


EXCLUDE = {"Anger","BPD","EatingDisorders","MMFB","StopSelfHarm","SuicideWatch","addiction","alcoholism",\
			"depression","feelgood","getting_over_it","hardshipmates","mentalhealth","psychoticreddit",\
			"ptsd","rapecounseling","schizophrenia","socialanxiety","survivorsofabuse","traumatoolbox"}
TOTAL_LIWC = 18
TRAINFS = ['../umd_reddit_suicidewatch_dataset/reddit_posts/controls/split_80-10-10/TRAIN.txt', 'umd_reddit_suicidewatch_dataset/reddit_posts/sw_users/split_80-10-10/TRAIN.txt']
TESTFS = ['../umd_reddit_suicidewatch_dataset/reddit_posts/controls/split_80-10-10/TEST.txt','umd_reddit_suicidewatch_dataset/reddit_posts/sw_users/split_80-10-10/TEST.txt']
DEVFS = ['../umd_reddit_suicidewatch_dataset/reddit_posts/controls/split_80-10-10/DEV.txt','umd_reddit_suicidewatch_dataset/reddit_posts/sw_users/split_80-10-10/DEV.txt']
ANNOFS = ['../umd_reddit_suicidewatch_dataset/reddit_annotation/crowd.csv','umd_reddit_suicidewatch_dataset/reddit_annotation/expert.csv']

#[postid, userid, timestamp, subreddit]

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
	return userDict,mentalHealthVec,subredditVecDict



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

'''post from TEXT FILE
  RAW POST

  [post_id]
  [user_id]
  [timestamp]
  [subreddit]
  [post_title]
  [post_body]'''

def prepare():
	#If done with process unpickle
	if os.path.exists('trainingData.p') and os.path.exists('testData.p') and os.path.exists(
			'devData.p') and os.path.exists('devTestData.p'):
		with open("trainingData.p", "rb") as f:
			trainPosts = pickle.load(f)
		with open("testData.p", "rb") as f:
			testPosts = pickle.load(f)
		with open("devData.p", "rb") as f:
			devPosts = pickle.load(f)
		with open("devTestData.p", "rb") as f:
			devTestPosts = pickle.load(f)

	#else go through each piece
	else:
		#part A
		if os.path.exists('allText.p') and os.path.exists('allPosts.p') and os.path.exists('suicideTimes.p'):
			with open('allText.p', 'rb') as f:
				allText = pickle.load(f)
			with open('allPosts.p', 'rb') as f:
				allPosts = pickle.load(f)
			with open('suicideTimes.p', 'rb') as f:
				allSuicideTimes = pickle.load(f)
		else:
			allText, allPosts, allSuicideTimes = _processDataset(['../umd_reddit_suicidewatch_dataset/reddit_posts/*/*.posts'],'liwc.p')

		if os.path.exists('docTopicVecs.p'):
			with open('docTopicVecs.p', 'rb') as f:
				docTopicVecs = pickle.load(f)
				ntopics = len(docTopicVecs[0])
		else:
			docTopicVecs, ntopics = _allText2TopicModel(allText, 'engStops')

		if os.path.exists('userDict.p') and os.path.exists('mentalHealthVec.p') and os.path.exists('subredditVecs.p'):
			with open('userDict.p', 'rb') as f:
				userDict = pickle.load(f)
			with open('mentalHealthVec.p', 'rb') as f:
				mentalHealthVec = pickle.load(f)
			with open('subredditVecs.p', 'rb') as f:
				subredditVecDict = pickle.load(f)
		else:
			userDict, mentalHealthVec, subredditVecDict = _addTopicVectorDataAndGroupByUser(docTopicVecs, ntopics, allPosts)

		allocator = makeAllocationDict(TRAINFS, TESTFS, DEVFS, ANNOFS)
		trainPosts, testPosts, devPosts, devTestPosts = _interpretFeatsAndAllocate(userDict, mentalHealthVec, subredditVecDict, allSuicideTimes, ntopics, allocator)

	return trainPosts, testPosts, devPosts, devTestPosts



