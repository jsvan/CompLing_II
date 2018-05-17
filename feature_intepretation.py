

# idx:  0       1                       2               3           4       5           6
# post: [userid,subreddit,              totw,           totmissp,   tot1sg, totpron,    totpres,
#
# ...   7       8 - 25                   26             27          28      29          30
# ...   totvrb, [funcwrdcts and liwc],  [topicSpaceVec],wkday,      hr,     timestamp,  label]






def interpretFeatures(bucket, dicSub2TopVec, mentalHealthVec):
	bucket_features=[] # only filled by bucket level features
	interpretted_bucket = [] #will fill with interpretted lists
	for post in bucket : #for every incoming post, first fill interp_buck with post lvel features. To be concat
		post_features = []
		fill post_features
		new bucket.append post features

	magic

	append bucket_features with post_features []+[]
	return bucket of concat. features

	'''
	:param bucket: a list(bucket) of posts of a list of features
	:param dicSub2TopVec:
	:param mentalHealthVec:
	:return: interpretted_post: a list of updated features
	'''



def _interpret_single_post(post, dicSub2TopVec, mentalHealthVec):
	#CHANGE 1:
	#   person_pronoun / all_pronoun
	#CHANGE 2:
	#   present_tense_verb / all_verb_tenses
	#CHANGE 3:
	#   %vocab from liwc
	#   words_in_anger/total_words
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