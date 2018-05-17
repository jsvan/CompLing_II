from unittest import TestCase
import user_week_buckets as b


TWO_WEEKS = 1209600

#       0       1        2    3        4      5       6
#post: [userid,subreddit,totw,totmissp,tot1sg,totpron,totpres,
#    7       8                      9             10    11 12       13
# ...totvrb,[funcwrdcts and liwc],[topicSpaceVec],wkday,hr,timestamp,label]

class Test_create_suicide_bucket(TestCase):
	def test__create_suicide_bucket(self):
		userList=[]
		post = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		userList.append(post)
		post = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
		userList.append(post)
		post = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0] #not suicide bucket
		userList.append(post)
		post = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, TWO_WEEKS+TWO_WEEKS, 0] #suicide bucket
		userList.append(post)
		post = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, TWO_WEEKS+TWO_WEEKS+1, 0]
		userList.append(post)
		post = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, TWO_WEEKS+TWO_WEEKS+2, 0]
		userList.append(post)
		post = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, TWO_WEEKS+TWO_WEEKS+TWO_WEEKS+TWO_WEEKS, 0]
		userList.append(post)
		post = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, TWO_WEEKS+TWO_WEEKS+TWO_WEEKS+TWO_WEEKS+7, 0]
		userList.append(post)

		suicideList={}
		user = []
		user.append(TWO_WEEKS+TWO_WEEKS+1)
		user.append(TWO_WEEKS+TWO_WEEKS+TWO_WEEKS+TWO_WEEKS+2)
		suicideList[0]=(user)

		truck = b.bucket_user_list(userList, suicideList)
		for i in truck:
			print(i)

		self.assertEqual(1,1)
