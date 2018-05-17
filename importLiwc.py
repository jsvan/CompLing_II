importantWords = set(['anger', 'health', 'sexual', 'money', 'death', 'friend', 'family', 'sad', 'keep', 'track', 'of', 'several', 'subword', 'categories', 'verb','auxverb','past','present','future','adverb','prep','conj','negate','quant','number'])
import pickle
with open('./other_materials/liwc/LIWC2007.dic', 'r') as f:
  fullLiwcText=f.read().split('\r\n')

for i in range(len(fullLiwcText)):
  fullLiwcText[i] = fullLiwcText[i].split('\t')

numberMap={} #old to new
count=0
i=1
categories={}
while fullLiwcText[i][0].isdigit() and len(fullLiwcText[i])==2:
  #IS CATEGORY
  if fullLiwcText[i][1] in importantWords:
    numberMap[fullLiwcText[i][0]] = count 
    categories[numberMap[fullLiwcText[i][0]]] = fullLiwcText[i][1]
    count+=1
  i+=1
  #{2 -> 'categ'
  # 7 -> 'categ2'}

liwcDic={}
for liwc in fullLiwcText[i:]:
  for number in liwc[1:]:
    if number in numberMap:
      if liwc[0] in liwcDic:
        liwcDic[liwc[0]].append(numberMap[number])
      else:
        liwcDic[liwc[0]] = [numberMap[number]]
#print(liwcDic)
#print('CATEGORIES')
print(categories)
pickle.dump(liwcDic, open("liwc.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(categories, open("liwcKey.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
