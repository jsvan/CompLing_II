

# idx:  0       1                       2               3           4       5           6
# post: [userid,subreddit,              totw,           totmissp,   tot1sg, totpron,    totpres,
#
# ...   7       8 - 24                   25             26          27      28          29
# ...   totvrb, [funcwrdcts and liwc],  [topicSpaceVec],wkday,      hr,     timestamp,  label]

# LIWC CATEGORIES
#{0: 'verb', 1: 'auxverb', 2: 'past', 3: 'present', 4: 'future', 5: 'adverb', 6: 'conj', 7: 'negate', 8: 'quant',
# 9: 'number', 10: 'family', 11: 'anger', 12: 'sad', 13: 'health', 14: 'sexual', 15: 'money', 16: 'death'}
w = {
	'user_id': 0,
	'subreddit': 1,
	'totw': 2,
	'totmissp':3,
	'tot1sg':4,
	'totpron':5,
	'totpres':6,
	'totvrb':7,
	'liwc_verb':8,
	'top_space_vec':26,
	'wkday':27,
	'hr':2,
	'timestamp':28,
	'label':29
	}




def interpretFeatures(bucket, dicSub2TopVec, mentalHealthVec):
	'''
	:param bucket: a list(bucket) of posts of a list of features
	:param dicSub2TopVec:
	:param mentalHealthVec:
	:return: interpretted_post: a list of updated features
	'''

	interpretted_bucket = [] #will fill with interpretted lists
	for post in bucket : #for every incoming post, first fill interp_buck with post lvel features. To be concatd
		post_features = _interpret_bucket(post, dicSub2TopVec, mentalHealthVec)
		interpretted_bucket.append(post_features)

	bucket_features = _interpret_bucket(bucket, dicSub2TopVec, mentalHealthVec)

	for i in range(len(interpretted_bucket)):
		interpretted_bucket[i]=interpretted_bucket[i] + bucket_features

	return interpretted_bucket





def _interpret_single_post(p, dicSub2TopVec, mentalHealthVec):
	interpretted_post=[]

	#CHANGE 1:
	#   person_pronoun / all_pronoun
	interpretted_post.append(float(p[w['tot1st']]) / float(p[w['totpron']]))

	#CHANGE 2:
	#   present_tense_verb / all_verb_tenses
	interpretted_post.append(float(p[w['totpres']]) / float(p[w['totvrb']]))

	#CHANGE 3:
	#   %vocab from liwc
	#   words_in_anger/total_words
	interpretted_post.append(float(p[w['']]) / float(p[w['f_words_liwc']]))

		#CHANGE 4:
		#   %vocab from liwc
		#   words_in_sad/total_words
		#CHANGE 5:
		#   %vocab from liwc
		#   words_in_health/total_words
		#CHANGE 6:
		#   %vocab from liwc
		#   words_in_sexual/total_words
		#CHANGE 7:
		#   %vocab from liwc
		#   words_in_money/total_words
	    #CHANGE 8:
		#   %vocab from liwc
		#   words_in_death/total_words
	    #CHANGE 9:
		#   %vocab from liwc
		#   words_in_friednds/total_words
	    #CHANGE 10:
		#   %vocab from liwc
		#   words_in_family/total_words
	#CHANGE 11:
	#   degree ling accomm dunno how
	#   cos sim (first half liwc cat to funct words), (liwc func words subreddit)
	#CHANGE 12:
	#   READABILITY
	#CHANGE 13:
	#   mental health themacity
	#   cosine sim (topic vec post), (topic vec mental health)
	#CHANGE 14:
	#   cosine sim (topic vec post), (topic vec subreddit)
	#CHANGE 15:
	#   Spelling accuracy
	#   number misspelled / total words post
	return intepretted_post

def _interpret_bucket(post, dicSub2TopVec, mentalHealthVec):
	#CHANGE 1:
	#   Time Dist Posts
	#       vector of eight probabilities: one for each TOD x WKT
	#CHANGE 2:
	#   Post Frequency
	#   	total posts in bucket; an int
	#CHANGE 3:
	#   Avg Post Length
	#       Sum(totalWords)/Total posts in bucket
	return interpretted_bucket