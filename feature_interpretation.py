import numpy as np
import scipy.spatial.distance as distance

# idx:  0       1                       2               3           4       5           6
# post: [userid,subreddit,              totw,           totmissp,   tot1sg, totpron,    totpres,
#
# ...   7       8 - 25                   26             27          28      29          30
# ...   totvrb, [funcwrdcts and liwc],  [topicSpaceVec],wkday,      hr,     timestamp,  label]

# LIWC CATEGORIES
# {0: 'verb', 1: 'auxverb', 2: 'past', 3: 'present', 4: 'future', 5: 'adverb', 6: 'conj', 7: 'negate', 8: 'quant',
# 9: 'number', 10: 'family', 11: 'friend', 12: 'anger', 13: 'sad', 14: 'health', 15: 'sexual', 16: 'money', 17: 'death'}
w = {
	'user_id': 0,
	'subreddit': 1,
	'totw': 2,
	'totmissp': 3,
	'tot1sg': 4,
	'totpron': 5,
	'totpres': 6,
	'totvrb': 7,

	# function words/style related
	'liwc_v': 8,
	'liwc_aux_v': 9,
	'liwc_past': 10,
	'liwc_prsnt': 11,
	'liwc_futr': 12,
	'liwc_adv': 13,
	'liwc_conj': 14,
	'liwc_neg': 15,
	'liwc_quant': 16,
	'liwc_num': 17,

	# thematic
	'liwc_fam': 18,
	'liwc_friend': 19,
	'liwc_anger': 20,
	'liwc_sad': 21,
	'liwc_health': 22,
	'liwc_sex': 23,
	'liwc_money': 24,
	'liwc_death': 25,

	'top_space_vec': 26,
	'wkday': 27,
	'hr': 28,
	'timestamp': 29,
	'label': 30
}


#   cos sim (first half liwc cat to funct words), (liwc func words subreddit)


def cos_sim(a, b):
	a = np.array(a)
	b = np.array(b)
	return 1.0 - distance.cosine(a, b)


def day_time_probs(bucket, n):
	t_buck = np.array(bucket).T
	weekends = t_buck[0][:]
	day_quartile = t_buck[1][:]
	day_time_buckets = [0] * 8
	weekends = 4 * weekends
	for post_idx in range(n):
		day_time_buckets[weekends[post_idx] + day_quartile[post_idx]] += (1.0 / float(n))
	return day_time_buckets


def interpretFeatures(bucket, dicSub2TopVec, mentalHealthVec, ntopics):
	'''
	:param bucket: a list(bucket) of posts of a list of features
	:param dicSub2TopVec:
	:param mentalHealthVec:
	:return: interpretted_post: a list of updated features
	'''
	# [ dayTime x 8, nposts, avgPostLen, missp%, Liwc%s, vrbRatio, pronRatio, cosSims]
	out = list()
	n = len(bucket)

	out += day_time_probs([post[27:29] for post in bucket], n)

	summedVec = getSums(bucket, range(2, 26))
	totw = summedVec[0]

	out.append(n)
	out.append(totw / n)
	out.append(summedVec[1] / totw)
	out += [summedVec[i] / totw for i in range(16, 24)]
	out.append(summedVec[4] / summedVec[5])
	out.append(summedVec[2] / summedVec[3])

	topicVecs = [(vec[1], vec[26]) for vec in bucket]
	funcVecs = [(vec[1], vec[8:18]) for vec in bucket]
	out.append(sumSimilarity(topicVecs, dicSub2TopVec, 19, 19 + ntopics) / n)
	out.append(sumSimilarity([(name, [val / totw for val in vec]) for name, vec in funcVecs], dicSub2TopVec, 0, 10) / n)
	out.append(cos_sim(getSums((vec[1] for vec in topicVecs), ntopics), mentalHealthVec))

	return out


def getSums(bucket, idxs):
	return [sum(vec[i] for vec in bucket) for i in idxs]


def sumSimilarity(vecs, subredditDict, startIdx, stopIdx):
	return sum(cos_sim(vec[1], subredditDict[vec[0]][startIdx:stopIdx]) for vec in vecs)


	# def _interpret_single_post(p, dicSub2TopVec, mentalHealthVec):
	# 	interpretted_post=[]

	# 	interpretted_post.append(float(p[w['tot1st']]) / float(p[w['totpron']]))
	# 	interpretted_post.append(float(p[w['totpres']]) / float(p[w['totvrb']]))
	# 	interpretted_post.append(float(p[w['liwc_anger']]) / float(p[w['totw']]))
	# 	interpretted_post.append(float(p[w['liwc_sad']]) / float(p[w['totw']]))
	# 	interpretted_post.append(float(p[w['liwc_health']]) / float(p[w['totw']]))
	# 	interpretted_post.append(float(p[w['liwc_sex']]) / float(p[w['totw']]))
	# 	interpretted_post.append(float(p[w['liwc_money']]) / float(p[w['totw']]))
	# 	interpretted_post.append(float(p[w['liwc_death']]) / float(p[w['totw']]))
	# 	interpretted_post.append(float(p[w['liwc_friend']]) / float(p[w['totw']]))
	# 	interpretted_post.append(float(p[w['liwc_fam']]) / float(p[w['totw']]))
	# 	#   cos sim (first half liwc cat to funct words), (liwc func words subreddit)
	# 	interpretted_post.append(cos_sim([v/p[w["totw"]] for v in p[8:18]], dicSub2TopVec[p[w["subreddit"]]][:10]))
	# 	#   cosine sim (topic vec post), (topic vec mental health)
	# 	interpretted_post.append(cos_sim(p[w["topic_space_vec"]], mentalHealthVec))
	# 	#   cosine sim (topic vec post), (topic vec subreddit)
	# 	interpretted_post.append(cos_sim(p[w["topic_space_vec"]], dicSub2TopVec[p[w["subreddit"]]][18:]))
	# 	#   Spelling accuracy
	# 	interpretted_post.append(p[w["totmissp"]] / p[w["totw"]])

	# 	return interpretted_post

	# def _interpret_bucket(bucket, dicSub2TopVec, mentalHealthVec):
	# 	num_posts=len(bucket)
	# 	interpretted_bucket=[]
	# 	#   CHANGE 1:
	# 	#       Time Dist Posts
	# 	#       vector of eight probabilities: one for each TOD x WKT
	# 	interpretted_bucket += (day_time_probs(bucket))
	# 	#   CHANGE 2:
	# 	#       Post Frequency
	# 	#   	total posts in bucket; an int
	# 	interpretted_bucket.append(num_posts)
	# 	#   CHANGE 3:
	# 	#       Avg Post Length
	# 	#       Sum(totalWords)/Total posts in bucket
	# 	interpretted_bucket.append(float(sum([bucket[i][2] for i in range(num_posts)])) / float(num_posts))

	# 	return interpretted_bucket