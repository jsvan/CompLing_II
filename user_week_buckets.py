import feature_intepretation as feat

TWO_WEEKS = 1209600
MIN = -99999

def _is_sorted(l):
	p = MIN
	for i in l:
		if i < p:
			return False
		p=i
	return True

#labels in suicide bucket keep same
#labels outside suicide bucket keep zero
#       0       1        2    3        4      5       6
#post: [userid,subreddit,totw,totmissp,tot1sg,totpron,totpres,
#    7       8                      9             10    11 12       13
# ...totvrb,[funcwrdcts and liwc],[topicSpaceVec],wkday,hr,timestamp,label]

def _create_suicide_bucket(userList, suicideList, dicSub2TopVec, mentalHealthVec):
	'''
	:param userList:
	:return: list (truck) of lists (buckets) of lists (posts) of strings
	'''
	suicideTime=0
	begin_of_two_weeks = userList[0][13]
	end_of_two_weeks=userList[0][13]
	truck=[]
	bucket=[]
	current_timestamp=0
	user_id = userList[0][0]

	if not _is_sorted(suicideList):
		suicideList.sort()
	suicideList.reverse()

	for post in userList:
		if post[0] != user_id:
			raise "Multiple users found in bucket. Expected "+ user_id+ " but found "+ post[0]
		current_timestamp=post[12]

		if current_timestamp < end_of_two_weeks:
			bucket.append(post)
		else:
			if bucket:
				truck.append(bucket)
			begin_of_two_weeks=current_timestamp
			end_of_two_weeks = begin_of_two_weeks+TWO_WEEKS
			bucket=[]
			bucket.append(post)

		while suicideTime < begin_of_two_weeks and suicideList:
			suicideTime=suicideList.pop()

		if begin_of_two_weeks < suicideTime and suicideTime < end_of_two_weeks: #this bucket had at least one SW post
			post[13] = 1

	if bucket:
		truck.append(bucket)

	return truck

def _create_safe_bucket(userList, dicSub2TopVec, mentalHealthVec):
	'''

	:param userList:
	:return: list (truck) of lists (buckets) of lists (posts) of strings
	'''
	end_of_two_weeks = userList[0][13] + TWO_WEEKS
	truck = []
	bucket = []
	current_timestamp = 0
	user_id = userList[0][0]

	for post in userList:
		if post[0] != user_id:
			raise "Multiple users found in bucket. Expected " + user_id + " but found " + post[0]

		current_timestamp = post[12]
		interpretted_post = feat.interpretFeatures(post, dicSub2TopVec, mentalHealthVec)
		if current_timestamp < end_of_two_weeks:
			bucket.append(interpretted_post)
		else:
			if bucket:
				truck.append(bucket)
			end_of_two_weeks = current_timestamp + TWO_WEEKS
			bucket = []
			bucket.append(interpretted_post)

	if bucket:
		truck.append(bucket)

	return truck

#userList: [post, post, post]
#       0       1        2    3        4      5       6
#post: [userid,subreddit,totw,totmissp,tot1sg,totpron,totpres,
#    7       8                      9             10    11 12       13
# ...totvrb,[funcwrdcts and liwc],[topicSpaceVec],wkday,hr,timestamp,label]
def interpret_post_features_by_user(userList, suicideDic, dicSub2TopVec, mentalHealthVec):
	'''
	:param userList: list for a single user of their posts
	:param suicideDic: dic of user to list of timestamp of every time posted in suicide watch
	:return: list of buckets (list) of posts, with labels changed
	'''
	u = userList[0][0]
	if u in suicideDic:
		return _create_suicide_bucket(userList, suicideDic[userList[0][0]], dicSub2TopVec, mentalHealthVec)
	else:
		return _create_safe_bucket(userList, dicSub2TopVec, mentalHealthVec)


