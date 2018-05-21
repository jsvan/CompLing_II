from nltk import word_tokenize, pos_tag
from autocorrect import spell
from data_utils import *
from glob import glob
from sys import argv
import pickle

EXCLUDE = {"Anger", "BPD", "EatingDisorders", "MMFB", "StopSelfHarm", "SuicideWatch", "addiction", "alcoholism", \
           "depression", "feelgood", "getting_over_it", "hardshipmates", "mentalhealth", "psychoticreddit", \
           "ptsd", "rapecounseling", "schizophrenia", "socialanxiety", "survivorsofabuse", "traumatoolbox"}
DATAFILES = [["../umd_reddit_suicidewatch_dataset/reddit_posts/*/*1.posts"], \
             ["../umd_reddit_suicidewatch_dataset/reddit_posts/*/*2.posts"], \
             ["../umd_reddit_suicidewatch_dataset/reddit_posts/*/*3.posts"], \
             ["../umd_reddit_suicidewatch_dataset/reddit_posts/*/*4.posts"], \
             ["../umd_reddit_suicidewatch_dataset/reddit_posts/*/*5.posts",
              "../umd_reddit_suicidewatch_dataset/reddit_posts/*/*0.posts"], \
             ["../umd_reddit_suicidewatch_dataset/reddit_posts/*/*6.posts",
              "../umd_reddit_suicidewatch_dataset/reddit_posts/*/*9.posts"], \
             ["../umd_reddit_suicidewatch_dataset/reddit_posts/*/*7.posts",
              "../umd_reddit_suicidewatch_dataset/reddit_posts/*/*8.posts"]]


def _processDataset(idx, liwcFile):
	'''
	:param dataFiles:
	:param liwcFile:
	:param stopFile:
	:return: pickles post data object
	:return: pickles all the text as a list of tokenized words
	:return: pickles suicide times dict
	'''
	print("A")
	with open(liwcFile, "rb") as lfile:
		liwc = pickle.load(lfile)
	dataFilenames = list()
	for ptrn in DATAFILES[idx]:
		dataFilenames += glob(ptrn)
	msDict = {}
	allText = list()
	allPosts = list()
	suicideTimes = dict()
	for dataFile in dataFilenames:
		with open(dataFile, "rU", errors="surrogateescape") as data:
			count = 0
			for post in data:  # post string, a line from file
				if count % 500 == 0:
					print(dataFile, count)
				count += 1
				# print('*', end='', flush=True)
				post = post.strip().split("\t")
				if len(post) > 4:  # post a list of strings (post info)
					titleLast = post[4][-1:]
					if titleLast.isalnum():  # i.e. not a punctuation mark:
						post[4] += "."
					post = post[:4] + [" ".join(post[4:])]
					subreddit = post[3]
					if subreddit in EXCLUDE:
						allText += [spellcheck(wrd.lower(), False, msDict) for wrd in word_tokenize(post[4])]
						allText.append("$|$")
						allPosts.append("IGNORE")
						if subreddit == "SuicideWatch":
							suicideTimes[post[1]] = suicideTimes.get(post[1], list()) + [int(post[2])]
					else:
						features = [0] * 31
						features[0] = post[1]
						features[-2] = int(post[2])
						features[1] = subreddit
						features = _processPostText(post[4], allText, msDict, liwc, features)
						weekend, daytime = timeToDate(int(post[2]))
						features[-4] = weekend
						features[-3] = daytime
						allPosts.append(features)
		print(dataFile, ' ending.')
	print('Pickling')
	with open("allText_%d.p" % idx, "wb") as f:
		pickle.dump(allText, f)
	with open("allPosts_%d.p" % idx, "wb") as f:
		pickle.dump(allPosts, f)
	with open("suicideTimes_%d.p" % idx, "wb") as f:
		pickle.dump(suicideTimes, f)


# [userid,subreddit,totw,totmissp,tot1sg,totpron,totpres,totvrb,[funcwrdcts and liwc],[topicSpaceVec],wkday,hr,timestamp,label]
def _processPostText(post, docFile, msdict, liwcDict, featureList):
	wrdList = [spellcheck(wrd.lower(), featureList, msdict) for wrd in word_tokenize(post)]
	docFile += wrdList
	docFile.append("$|$")
	tags = pos_tag(wrdList)
	for wrd, tag in tags:
		if tag[0:1] == "V":
			featureList[7] += 1
			if tag in {"VBG", "VBP", "VBZ"}:
				featureList[6] += 1
		elif tag[0:3] == "PRP":
			featureList[5] += 1
			if wrd in {"me", "my", "I", "myself", "mine"}:
				featureList[4] += 1
		elif wrd in liwcDict:
			themes = liwcDict[wrd]
			for theme in themes:
				featureList[8 + theme] += 1
	return featureList


def spellcheck(wrd, lst, msdict):
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


if __name__ == "__main__":
	_processDataset(argv[1], 'liwc.p')

