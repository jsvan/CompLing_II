import feature_interpretation as feat

TWO_WEEKS = 1209600
MIN = -99999
TIMESTAMP_IDX = 29
LABEL_IDX = -1
USER_ID_IDX=0

def _is_sorted(l):
	p = MIN
	for i in l:
		if i < p:
			return False
		p=i
	return True


# idx:  0       1                       2               3           4       5           6
# post: [userid,subreddit,              totw,           totmissp,   tot1sg, totpron,    totpres,
#
# ...   7       8 - 25                   26             27          28      29          30
# ...   totvrb, [funcwrdcts and liwc],  [topicSpaceVec],wkday,      hr,     timestamp,  label]

# LIWC CATEGORIES
#{0: 'verb', 1: 'auxverb', 2: 'past', 3: 'present', 4: 'future', 5: 'adverb', 6: 'conj', 7: 'negate', 8: 'quant',
# 9: 'number', 10: 'family', 11: 'friend', 12: 'anger', 13: 'sad', 14: 'health', 15: 'sexual', 16: 'money', 17: 'death'}

def _create_suicide_bucket(userList, suicideList, dicSub2TopVec, mentalHealthVec, ntopics):
	'''
		labels in suicide bucket keep same
		labels outside suicide bucket keep zero

	:param userList:
	:return: list (truck) of lists (buckets) of lists (posts) of strings
	'''
	suicideTime=0
	begin_of_two_weeks = userList[0][TIMESTAMP_IDX]
	end_of_two_weeks=userList[0][TIMESTAMP_IDX]
	truck=[]
	bucket=[]
	current_timestamp=0
	user_id = userList[0][USER_ID_IDX]

	if not _is_sorted(suicideList):
		suicideList.sort()
	suicideList.reverse()

	for post in userList:
		if post[0] != user_id:
			raise "Multiple users found in bucket. Expected "+ user_id+ " but found "+ post[0]

		current_timestamp=post[TIMESTAMP_IDX]

		if current_timestamp < end_of_two_weeks:
			bucket.append(post)
		else:
			if bucket:
				truck.append(feat.interpretFeatures(bucket, dicSub2TopVec, mentalHealthVec, ntopics))
			begin_of_two_weeks=current_timestamp
			end_of_two_weeks = begin_of_two_weeks+TWO_WEEKS
			bucket=[]
			bucket.append(post)

		while suicideTime < begin_of_two_weeks and suicideList:
			suicideTime=suicideList.pop()

		if suicideTime < begin_of_two_weeks or end_of_two_weeks < suicideTime: #this bucket is NOT SW
			post[LABEL_IDX] = -1
	#   else:                                                                   #ths bucket is SW
	#      maintain label

	if bucket:
		truck.append(feat.interpretFeatures(bucket, dicSub2TopVec, mentalHealthVec, ntopics))

	return truck



def _create_safe_bucket(userList, dicSub2TopVec, mentalHealthVec,ntopics):
	'''

	:param userList:
	:return: list (truck) of lists (buckets) of lists (posts) of strings
	'''
	end_of_two_weeks = userList[0][TIMESTAMP_IDX] + TWO_WEEKS
	truck = []
	bucket = []
	current_timestamp = 0
	user_id = userList[0][USER_ID_IDX]

	for post in userList:
		if post[0] != user_id:
			raise "Multiple users found in bucket. Expected " + user_id + " but found " + post[0]

		current_timestamp = post[TIMESTAMP_IDX]
		if current_timestamp < end_of_two_weeks:
			bucket.append(post)
		else:
			if bucket:
				truck.append(feat.interpretFeatures(bucket, dicSub2TopVec, mentalHealthVec, ntopics))
			end_of_two_weeks = current_timestamp + TWO_WEEKS
			bucket = []
			bucket.append(post)

	if bucket:
		truck.append(feat.interpretFeatures(bucket, dicSub2TopVec, mentalHealthVec, ntopics))

	return truck


def interpret_post_features_by_user(userList, suicideDic, dicSub2TopVec, mentalHealthVec, ntopics):
	'''
	:param userList: list for a single user of their posts
	:param suicideDic: dic of user to list of timestamp of every time posted in suicide watch
	:return: list of buckets (list) of posts, with labels changed
	'''
	if len(userList[0]) != 31:
		raise "post wrong size, reevaluate. Thought 30, was " + len(userList[0]) +"\n"+userList[0]


	u = userList[0][USER_ID_IDX]

	userList.sort(key = lambda post: post[29])

	if u in suicideDic:
		truck= _create_suicide_bucket(userList, suicideDic[userList[0][USER_ID_IDX]], dicSub2TopVec, mentalHealthVec, ntopics)
	else:
		truck= _create_safe_bucket(userList, dicSub2TopVec, mentalHealthVec, ntopics)

	# 0 [ dayTime x 8,
	# 1
	# 2
	# 3
	# 4
	# 5
	# 6
	# 7
	# 8 nposts,
	# 9 avgPostLen,
	# 10 missp%,
	# 11 Liwc%s (same order as above),
	# vrbRatio,
	# \ pronRatio,
	# sim(subreddit,post),
	# sim(subredditstyle,poststyle),
	# sim(post,mentalhealth)]

	print(len(truck[0]))
	for i in range(len(truck[0])):
		print(i, ": ", truck[0][i])
		print(i,"-: ", truck[0][-i])
	steady_label = truck[0][LABEL_IDX]
	for post in truck:
		if post[LABEL_IDX] != steady_label:
			raise "Bucket with Unmatchin Labels Found with User "+ str(post[USER_ID_IDX])+". Label "+ str(post[LABEL_IDX])+" Does Not Match with Expected "+ str(steady_label)+ "."

	return truck
