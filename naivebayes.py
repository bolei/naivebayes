#!/usr/bin/python

import sys
import math

data_folder = "data"
cat_index = ["con", "lib"]

# model
prior = []
conditional_prob = []
vocab = []

def getunigram(corpus, vocab):
	unigram = {}
	for token in vocab:
		unigram[token] = 0
	for token in corpus:
		unigram[token] += 1
	return unigram

def trainModel(fileName):
	count_doc = []
	unigram_arr = []
	cat_tokens = []
	vocabset = set([])
	
	
	global prior
	global conditional_prob
	global vocab
	
	for i in xrange(len(cat_index)):
		count_doc.append(0)
		cat_tokens.append([])
		conditional_prob.append([])
		unigram_arr.append({})
		prior.append(0)
	
	with open(fileName) as fhandle:
		fileNames = fhandle.readlines()
		for fname in fileNames:
			label = cat_index.index(fname[0:3])
			with open(data_folder + '/' + fname.strip()) as f:
				content = f.readlines()
	
				count_doc[label] += 1
				cat_tokens[label].extend(content)
	
				for token in content:
					vocabset.add(token)
		
		example_count = sum(count_doc)
		vocab = list(vocabset)
		
		for i in xrange(len(prior)):
			for j in xrange(len(vocab)):
				conditional_prob[i].append(0)
		
		for i in xrange(len(count_doc)):
			prior[i] = float(count_doc[i]) / float(example_count)
			unigram_arr[i] = getunigram(cat_tokens[i], vocab)
			
			for w in vocab:
				conditional_prob[i][vocab.index(w)] = float(unigram_arr[i][w] + 1) / float(len(cat_tokens[i]) + len(vocabset))

def getLogLikelihood(category, corpus):
	ll = math.log(prior[category])
	for w in corpus:
		if(w in vocab):
			ll += math.log(conditional_prob[category][vocab.index(w)])
	return ll

def testModel(fileName):
	with open(fileName) as fHandle:
		fnames = fHandle.readlines()
		for fname in fnames:
			fname = fname.strip()
			with open(data_folder + '/' + fname.strip()) as f:
				content = f.readlines()
				max_ll = sys.float_info.min
				cat = 0
				for i in xrange(len(prior)):
					ll = getLogLikelihood(i, content)
					if max_ll < ll:
						max_ll = ll
						cat = i
				classification = cat_index[cat].upper()[0]
			print fname + " " + classification

def main():
	split_train = sys.argv[1]
	split_test = sys.argv[2]
	trainModel(split_train)
	testModel(split_test)
	

if __name__ == "__main__":
	main()
