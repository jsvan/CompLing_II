# random exp
from multiprocessing.pool import ThreadPool
from glob import glob
from nltk import pos_tag

files = glob("../umd_reddit_suicidewatch_dataset/reddit_posts/*/*.posts")
thread_pool = ThreadPool(processes=7)

result = thread_pool.map(randoFunc,files)

def randoFunc(filename):
	bigList = list()
	with open(filename,"rU") as f:
		for ln in f:
			bigList += pos_tag(ln.strip.split())
	return bigList
