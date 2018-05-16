import nltk 


trainSet = pickle.load(open("train_set.p"), "rb")
testSet = pickle.load(open("train_set.p"), "rb")
trainSet = pickle.load(open("train_set.p"), "rb")


#0  [post_id]
#1  [user_id]
#2  [timestamp]
#3  [subreddit]
#4  [post_title]
#5  [post_body]



for post in trainSet:
  post[5] = nltk.tokenize(post[5])
  post.append(len(post[5])


userToAvg = {}

name = ''
length = 0.0
count = 1.0
for i in trainSet:
  if name != i[1]:
    userToAvg[name] = length / count
    length = 0.0
    count = 0.0
    name = i[1]
  length += i[5].count(' ')
  count +=1.0t
userToAvg[name] = length / count

for post in trainSet:
  post.append(userToAvg[post[1]])

userToAvg = {}



