from nltk import word_tokenize,pos_tag,download
from sys import argv
from autocorrect import spell
from glob import glob
import pickle
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
#	print(dataFiles)
	with open(liwcFile,"rb") as lfile:
		liwc = pickle.load(lfile)
	msDict = dict()	
	dataFilenames = list()
	# suicideTimes = dict()
	for dataFilePtrn in dataFiles:
#		print(dataFilePtrn)
		dataFilenames += glob(dataFilePtrn)
#	print(dataFilenames)
	for dataFile in dataFilenames:
		print(dataFile)
		with open(dataFile,"rU",errors="surrogateescape") as data:
			allText = list()
			allPosts = list()
			suicideTimes = dict()
			for post in data: #post string, a line from file
				print('*', end='', flush=True)
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
			with open(dataFile+"_allText.p", "wb") as f:
				pickle.dump(allText, f)
			with open(dataFile+"_allPosts.p", "wb") as f:
				pickle.dump(allPosts, f)
			with open(dataFile+"_suicideTimes.p", "wb") as f:
				pickle.dump(suicideTimes, f)

def _allText2TopicModel(allocationDict, allPosts, allText, suicideTimes, stopFile):
	print('B')
	model = collocateAndLDA(allText,stopFile)
	ntopics = len(docTopicVecs[0])

def _nextStuff():
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
			longVec = post[8:8+TOTAL_LIWC]+docTopicVecs[idx]
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


	print('C')
	for (subreddit, (vec,n,w)) in subredditVecDict.items():
		subredditVecDict[subreddit] = [vec[i]/w for i in range(TOTAL_LIWC)]+[vec[i]/n for i in range(TOTAL_LIWC,longVeclen)]

	for postList,fname in ((trainPosts,"train.p"),(devPosts,"dev.p"),(devTestPosts,"devTest.p")):
		userPostDict = dict()
		for post in postList:
			userPostDict[post[0]] = userPostDict.get(post[0],list()) + [post]
		outLabelled = list()
		outUnlabelled = list()
		for user in userPostDict:
			if allocationDict[user][1] == 0:
				bucketList = uwb.interpret_post_features_by_user(userPostDict[user], suicideTimes, subredditVecDict, mentalHealthVec,ntopics)
				outUnlabelled.append([bucket for bucket in bucketList if bucket[0][-1] == 0])
				outLabelled.append([bucket for bucket in bucketList if bucket[0][-1] == -1])
			else:
				outLabelled.append(uwb.interpret_post_features_by_user(userPostDict[user], suicideTimes, subredditVecDict, mentalHealthVec,ntopics))
		outTup = (outLabelled,outUnlabelled)
		with open(fname,"wb") as f:
			pickle.dump(outTup,f)

	userPostDict = dict()
	print('D')
	for post in testPosts:
		userPostDict[post[0]] = userPostDict.get(post[0],list()) + [post]
	outList = [uwb.interpret_post_features_by_user(userList,suicideTimes,subredditVecDict,mentalHealthVec) for user,userList in userPostDict.items()]
	print('Pickling')
	with open("test.p","wb") as f:
		pickle.dump(outList,f)

	with open("mentalHealthVec.p","wb") as tp:
		pickle.dump(mentalHealthVec,tp)
	with open("subredditVecs.p","wb") as tp:
		pickle.dump(subredditVecDict,tp)
	with open("suicideTimes.p","wb") as tp:
		pickle.dump(suicideTimes,tp)

	### DONE WITH PIPELINE?


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


