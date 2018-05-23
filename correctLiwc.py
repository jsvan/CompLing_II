import pickle

with open("liwc.p","rb") as f:
	x = pickle.load(f)

newLiwc = dict()
for wrd,lst in x.items():
	if wrd[-1] == "*":
		print(wrd)
		wrd =wrd[:-1]
		new = input("New form: ")
		while new != "":
			newLiwc[new.replace(".",wrd)] = lst
			new = input("New form: ")
	else:
		newLiwc[wrd] = lst

with open("liwc.p","wb") as f:
	pickle.dump(newLiwc,f)
