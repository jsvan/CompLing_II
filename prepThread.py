from nltk import word_tokenize
from nltk import pos_tag, download
from multiprocessing.pool import ThreadPool

#l is input list of posts
#tp.map(f, l)

from autocorrect import spell
from glob import glob
import pickle
import user_week_buckets as uwb

from data_utils import *

download("averaged_perceptron_tagger")
download("punkt")
# PATH_TO_STANF = "/home/ubuntu/stanford-postagger-full-2018-02-27/models/english-caseless-left3words-distsim.tagger"
# PATH_TO_JAR = "/home/ubuntu/stanford-postagger-full-2018-02-27/stanford-postagger-3.9.1.jar"
EXCLUDE = {"Anger","BPD","EatingDisorders","MMFB","StopSelfHarm","SuicideWatch","addiction","alcoholism",\
			"depression","feelgood","getting_over_it","hardshipmates","mentalhealth","psychoticreddit",\
			"ptsd","rapecounseling","schizophrenia","socialanxiety","survivorsofabuse","traumatoolbox"}
TOTAL_LIWC = 18
THREAD_COUNT = 8
TRAINFS = ['umd_reddit_suicidewatch_dataset/reddit_posts/controls/split_80-10-10/TRAIN.txt', 'umd_reddit_suicidewatch_dataset/reddit_posts/sw_users/split_80-10-10/TRAIN.txt']
TESTFS = ['umd_reddit_suicidewatch_dataset/reddit_posts/controls/split_80-10-10/TEST.txt','umd_reddit_suicidewatch_dataset/reddit_posts/sw_users/split_80-10-10/TEST.txt']
DEVFS = ['umd_reddit_suicidewatch_dataset/reddit_posts/controls/split_80-10-10/DEV.txt','umd_reddit_suicidewatch_dataset/reddit_posts/sw_users/split_80-10-10/DEV.txt']
ANNOFS = ['umd_reddit_suicidewatch_dataset/reddit_annotation/crowd.csv','umd_reddit_suicidewatch_dataset/reddit_annotation/expert.csv']
thread_pool = ThreadPool(processes=THREAD_COUNT)
msdict = dict()
liwc = dict()
# tagger = Tagger(PATH_TO_STANF,PATH_TO_JAR)


#[postid, userid, timestamp, subreddit]
def processDataset(dataFiles,liwcFile,stopFile):
	print('A')
	with open(liwcFile,"rb") as lfile:
		liwc = pickle.load(lfile)
	allocationDict = makeAllocationDict(TRAINFS, TESTFS, DEVFS, ANNOFS)
	allText = list()
	allPosts = list()
	dataFilenames = list()
	suicideTimes = dict()
	for dataFilePtrn in dataFiles:
		dataFilenames += glob(dataFilePtrn)

	res = thread_pool.map(delegate_file_to_threads, dataFilenames)  # treturn list of size file_count, of tuples (all_text_portion, all_posts_portion, suicide_times_portion)
	all_text_portions=[]
	all_posts_portions=[]
	all_suicide_portions={}
	for tuple in res:
		all_text_portions += tuple[0]
		all_posts_portions += tuple[1]
		all_suicide_portions.update(tuple[2])

	print('Pickling')
	with open("allText.p", "wb") as f:
		pickle.dump(allText, f)
	with open("allPosts.p", "wb") as f:
		pickle.dump(allPosts, f)
	with open("msdict.p", "wb") as f:
		pickle.dump(msdict, f)
	with open("suicideTimes.p", "wb") as f:
		pickle.dump(suicideTimes, f)

	print('B')
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
				bucketList = uwb.interpret_post_features_by_user(userPostDict[user], suicideTimes, subredditVecDict, mentalHealthVec)
				outUnlabelled.append([bucket for bucket in bucketList if bucket[0][-1] == 0])
				outLabelled.append([bucket for bucket in bucketList if bucket[0][-1] == -1])
			else:
				outLabelled.append(uwb.interpret_post_features_by_user(userPostDict[user], suicideTimes, subredditVecDict, mentalHealthVec))
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

#[userid,subreddit,totw,totmissp,tot1sg,totpron,totpres,totvrb,[funcwrdcts and liwc],[topicSpaceVec],wkday,hr,timestamp,label]
def processPostText(post, docFile, liwcDict, featureList):
	wrdList = [spellcheck(wrd.lower(),featureList) for wrd in word_tokenize(post)]
	if not wrdList:
		print(post)
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

def spellcheck(wrd,lst):
	global msdict
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



def delegate_file_to_threads(dataFile):
	# liwc
	all_text_portion = [] # wil lbe string, '$|$'
	all_posts_portion =[] # list of 1 element of either IGNORE or [features]
	suicide_times_portion = {} # dic from user id to post time?
	print(dataFile)
	count = 0
	with open(dataFile, "rU", errors="surrogateescape") as data:
		for post in data:  # post string, a line from file
			print('*', end='', flush=True)
			post = post.strip()
			if post:
				try:
					post = post.split("\t") #post a list of strings (post info)
					titleLast = post[4][-1:]
					if titleLast.isalnum(): #i.e. not a punctuation mark:
						post[4] += "."
					post[4] = " ".join(post[4:])
					subreddit = post[3]
					if subreddit in EXCLUDE:
						all_text_portion = [spellcheck(wrd.lower(),False) for wrd in word_tokenize(post[5])]
						all_text_portion.append("$|$")
						all_posts_portion.append("IGNORE")
						if subreddit == "SuicideWatch":
							suicide_times_portion[post[1]] = suicide_times_portion.get(post[1],list()) + [int(post[2])]
					else:
						features = [0]*31
						features[0] = post[1]
						features[-2] = int(post[2])
						features[1] = subreddit
						features = processPostText(post[4],all_text_portion,liwc,features)
						weekend, daytime = timeToDate(int(post[2]))
						features[-4] = weekend
						features[-3] = daytime
						all_posts_portion.append(features)
				except Exception as e:
					print(e)

			count +=1
	return (all_text_portion, all_posts_portion, suicide_times_portion)


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