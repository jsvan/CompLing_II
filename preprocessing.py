from nltk import word_tokenize
from nltk.tag import stanford as stanf

from autocorrect import spell
from glob import glob
import pickle
import user_week_buckets as uwb

from data_utils import *

EXCLUDE = {"Anger","BPD","EatingDisorders","MMFB","StopSelfHarm","SuicideWatch","addiction","alcoholism",\
			"depression","feelgood","getting_over_it","hardshipmates","mentalhealth","psychoticreddit",\
			"ptsd","rapecounseling","schizophrenia","socialanxiety","survivorsofabuse","traumatoolbox"}
TOTAL_LIWC = 18
TRAINFS = ['umd_reddit_suicidewatch_dataset/reddit_posts/controls/split_80-10-10/TRAIN.txt', 'umd_reddit_suicidewatch_dataset/reddit_posts/sw_users/split_80-10-10/TRAIN.txt']
TESTFS = ['umd_reddit_suicidewatch_dataset/reddit_posts/controls/split_80-10-10/TEST.txt','umd_reddit_suicidewatch_dataset/reddit_posts/sw_users/split_80-10-10/TEST.txt']
DEVFS = ['umd_reddit_suicidewatch_dataset/reddit_posts/controls/split_80-10-10/DEV.txt','umd_reddit_suicidewatch_dataset/reddit_posts/sw_users/split_80-10-10/DEV.txt']
ANNOFS = ['umd_reddit_suicidewatch_dataset/reddit_annotation/crowd.csv','umd_reddit_suicidewatch_dataset/reddit_annotation/expert.csv']

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
						allText += (spellcheck(wrd.lower(),False,msDict) for wrd in word_tokenize(post[5]))
						allText.append("$|$")
						allPosts.append("IGNORE")
						if subreddit == "SuicideWatch":
							suicideTimes[post[1]] = suicideTimes.get(post[1],list()) + [int(post[2])]
					else:
						features = [0]*31
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
				bucketList = uwb.interpret_post_features_by_user(userPostDict[user], suicideTimes, subredditVecDict, mentalHealthVec)
				outUnlabelled.append([bucket for bucket in bucketList if bucket[0][-1] == 0])
				outLabelled.append([bucket for bucket in bucketList if bucket[0][-1] == -1])
			else:
				outLabelled.append(uwb.interpret_post_features_by_user(userPostDict[user], suicideTimes, subredditVecDict, mentalHealthVec))
		outTup = (outLabelled,outUnlabelled)
		with open(fname,"wb") as f:
			pickle.dump(outTup,f)

	userPostDict = dict()
	for post in testPosts:
		userPostDict[post[0]] = userPostDict.get(post[0],list()) + [post]
	outList = [uwb.interpret_post_features_by_user(userList,suicideTimes,subredditVecDict,mentalHealthVec) for user,userList in userPostDict.items()]
	with open("test.p","wb") as f:
		pickle.dump(outList,f)

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






'''post from TEXT FILE
  RAW POST

  [post_id]
  [user_id]
  [timestamp]
  [subreddit]
  [post_title]
  [post_body]'''


if __name__ =='__main__':
	processDataset(["umd_reddit_suicidewatch_dataset/reddit_posts/controls/*.posts",
	                "umd_reddit_suicidewatch_dataset/reddit_posts/sw_users/*.posts"], "./liwc.p", "engStops")