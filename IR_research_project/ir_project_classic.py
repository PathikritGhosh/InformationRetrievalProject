__author__ = "Pathikrit Ghosh"
__version__ = "1.0.1"
__python_version___ = "2.7"

import shutil,os,sys
import math,string
import time
import itertools
import heapq
import nltk
import numpy as np
import cPickle as pickle
from sklearn import metrics
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from scipy.sparse import csr_matrix
from stemming.porter2 import stem
from nltk.corpus import stopwords
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer
from collections import Counter
from scipy.stats import norm
from scipy import sparse
from collections import Counter
from scipy.stats import norm

import warnings
warnings.filterwarnings("ignore")


cwd = os.getcwd()
'''
Place the Classic dataset (data files) on creating a folder by name "classic".
Create another folder "Classic_Dataset" and 4 sub folders by name "med", "cran", "cicsi", "cacm".
 
The hardcoded number(a, b, c, d) of files will be copied from the "classic" to corresponding folders ("cacm", "med", "cisi", "cran").

'''
dirPath = cwd + "/Classic_Dataset/med"
fileList = os.listdir(dirPath)
for fileName in fileList:
	os.remove(dirPath+"/"+fileName)

dirPath = cwd + "/Classic_Dataset/cran"
fileList = os.listdir(dirPath)
for fileName in fileList:
	os.remove(dirPath+"/"+fileName)

dirPath = cwd + "/Classic_Dataset/cisi"
fileList = os.listdir(dirPath)
for fileName in fileList:
	os.remove(dirPath+"/"+fileName)

dirPath = cwd + "/Classic_Dataset/cacm"
fileList = os.listdir(dirPath)
for fileName in fileList:
	os.remove(dirPath+"/"+fileName)

#CACM, CISI, CRAN, MED (total number of docs in corresponding classed : 3204, 1460, 1400, 1033)
a = 55
b = 45
c = 55
d = 50

#Hard code the "M" percentage value i.e.: the top 'M%' features you wish to select from each class.
#doc_top represents the top percentage of documents to be kept
#top_num represents the top percentage of features to be select by method1 (i.e. variance method)
#top_num represents the top percentage of features to be select by method2 (i.e. idf method)
top_num = int(sys.argv[1])
doc_top = 95
top_num2 = int(sys.argv[2])


# copy all the required documents to another folder from all the four classes
for j in range(0, a):
	dest = cwd + "/Classic_Dataset/cacm"
	num = "%.6d" % (j+1)
	name = "/classic/cacm." + (str)(num)
	source = cwd + name
	shutil.copy2(source, dest)

for j in range(0, b):
	dest = cwd + "/Classic_Dataset/cisi"
	num = "%.6d" % (j+1)
	name = "/classic/cisi." + (str)(num)
	source = cwd + name
	shutil.copy2(source, dest)

for j in range(0, c):
	dest = cwd + "/Classic_Dataset/cran"
	num = "%.6d" % (j+1)
	name = "/classic/cran." + (str)(num)
	source = cwd + name
	shutil.copy2(source, dest)

for j in range(0, d):
	dest = cwd + "/Classic_Dataset/med"
	num = "%.6d" % (j+1)
	name = "/classic/med." + (str)(num)
	source = cwd + name
	shutil.copy2(source, dest)

#Load The files/dataset
load_path = cwd + "/Classic_Dataset"
dataset=load_files(load_path, description=None, categories=None, load_content=True, shuffle=False, encoding=None, decode_error='strict', random_state=0)

#Class names and assigned numbers
class_names= list(dataset.target_names)

class_num = len(class_names)
class_numbers = []
for i in range(0, len(class_names)):
	class_numbers.append(i)

#Document class labels
d_labels = dataset.target

#Data from the dataset
vdoc = dataset.data

#Stemming the words
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stem(item))
    return stemmed

#Tokenizing each word
def tokenize(text):
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

#Count Tokenizer --> Finding doc-term frequency matrix
vec = CountVectorizer(tokenizer=tokenize, stop_words='english')
data = vec.fit_transform(vdoc).toarray()
voc = vec.get_feature_names()

#Finding Vocabulary
vocabulary = voc
voc_num = len(vocabulary)
doc_num = data.shape[0]

#final doc-term tfidf vector
tf_vec = TfidfTransformer(use_idf=True).fit(data)
vectors = tf_vec.transform(data)


idf_vec = TfidfTransformer(use_idf=False,smooth_idf=False).fit(data)
idf_vectors = idf_vec.transform(data)



# find centroid for documents of class a
def document_reduction(vectors,a,begin):
	'''
		this method takes the document vector and the starting position of the class in the vector and also the size of the class.
		then it finds the avg of the document vectors i.e. the centroid vector for the class.
		it computes the cosine similarity with each of the 'a' documents of the class and selects top (doc_top) percentage of documents from that class
		and returns sorted list of top documents of the class based on cosine similarity with the centroid		 
	'''
	
	#finding centroid
	centroid_doc = vectors[begin]
	for i in range((begin+1),a):
		centroid_doc+=vectors[i]

	centroid_doc /= a

	#find cosine similarity of class a centroid with all document vectors
	similarity = cosine_similarity(centroid_doc,vectors[begin:(begin+a)])
	
	#find top_documents
	top_documents =  heapq.nlargest(int(a*0.01*doc_top), zip(similarity[0], itertools.count(begin)))
	
	documents_to_keep = [item[1] for item in top_documents]
	return  sorted(documents_to_keep)

#class doc contains top documents from each class
class_doc = []
class_doc.append(document_reduction(vectors,a,0)) #document reduction being carried for each class'''
class_doc.append(document_reduction(vectors,b,a))
class_doc.append(document_reduction(vectors,c,(a+b)))
class_doc.append(document_reduction(vectors,d,(a+b+c)))


#all vocabulary in each document
cl_voc = []
for item in data:
	cl_voc.append([i for i, j in enumerate(item) if j != 0])

class_voc = []
for item in class_doc:
	temp = [cl_voc[x] for x in item]
	class_voc.append(list(set(list(itertools.chain.from_iterable(temp)))))



#class_vec and class_vec2 contains tf-idf anf idf vectors of each class
class_vec = []
class_vec2 = []
for i in range(0,len(class_doc)):
	s = []
	s2 = []
	for item in class_doc[i]:
		s.append(vectors.getrow(item).toarray().tolist()[0])
		
		sublist = []
		for item2 in range(len(vectors.getrow(item).toarray().tolist()[0])):
			if(idf_vectors.getrow(item).toarray().tolist()[0][item2]!=0):
				idf = vectors.getrow(item).toarray().tolist()[0][item2]/idf_vectors.getrow(item).toarray().tolist()[0][item2]
				#print idf_vectors.getrow(item).toarray().tolist()[0][item2],idf
				sublist.append(idf)
			else:
				sublist.append(0)
		s2.append(sublist)
	class_vec.append(csr_matrix(s))
	class_vec2.append(csr_matrix(s2))




def feature_select(vec,voc,vec2):
	'''this is the main feature selection method which taked two vectors tf-idf vector and idf vector for each class and returns (top_num*top_num2) features for the class'''
	
	
	'''
		first it finds the avg value of the tfidf for each term and then take its variance
		the top (top_num) percentage of terms are selected which have high variance
	'''
	avg_tfidf = []
	variance_list = []
	for item in voc:
		av_tf_idf = sum(vec.getcol(item).transpose().toarray()[0])/vec.shape[0]
		avg_tfidf.append(av_tf_idf)
		variance = 0
		temp = vec.getcol(item).transpose().toarray()[0]
		
		for item2 in temp:
			variance += (av_tf_idf - item2)**2

		variance = variance/float(len(temp))
		variance_list.append(variance)
		
	sel = heapq.nlargest(int(len(voc)*0.01*top_num), zip(variance_list, itertools.count()))
	sel_2 = [item[1] for item in sel]
	sel_indices = [voc[i] for i in sel_2]
	
	
	'''
		second it finds the the top (top_num2) percentage of terms from the earlier selected terms and returns those terms
	'''
	idf_list = []
	sel_indices2 = []
	for item in sel_indices:
		idf = vec2.getcol(item).transpose().toarray()[0][0]
		idf_list.append(idf)
	
	sel = heapq.nlargest(int(len(sel_indices)*0.01*top_num2), zip(idf_list, itertools.count()))
	sel_2 = [item[1] for item in sel]
	sel_indices2 = [sel_indices[i] for i in sel_2]
	#sel_indices2 are terms with top idf values
	return sel_indices2
	
	
#Sel_f_indices has all the vocabulary ranked (based on technique)
#taking the top (top_num *top_num2) features and re-building vectors i.e.: doc-term tfidf matrix
sel_f_indices = []
for i in range(0, len(class_voc)):
	sel_f_indices.append((feature_select(class_vec[i], class_voc[i],class_vec2[i])))
	
sel_f_indices = (list(set(list(itertools.chain.from_iterable(sel_f_indices)))))


temp = []
new_dlabel = []
for i in range(len(class_doc)):
	for item in class_doc[i]:
		temp.append(vectors.getrow(item).toarray().tolist()[0])
		new_dlabel.append(d_labels[item])

vector2 = csr_matrix(temp)

sel_f_indices.sort()
vector2 = vector2.transpose()	


s = []
for item in sel_f_indices:
	s.append(vector2.getrow(item).toarray().tolist()[0])
	
#new_vectors contains the matrix for the newly selected documents and features
new_vectors = csr_matrix(s)
new_vectors = new_vectors.transpose()

vector2 = vector2.transpose()



def Classification(X_train, X_test, y_train, y_test):
	#this is the classification technique that takes training and testing data and prints confusion matrix and precision and recall
	print "\n----------SVM linear Kernel--------\n"
	start1 = time.time()
	clf = svm.SVC(kernel='linear', C=1)
	clf.fit(X_train, y_train)
	print "Training time taken %f \n" % (time.time() - start1)

	start2 = time.time()
	y_pred = clf.predict(X_test).tolist()
	print "Testing time taken %f \n" % (time.time() - start2)
	print "Classification report:\n"
	print metrics.confusion_matrix(y_test,y_pred)
	print len(y_test)
	print metrics.accuracy_score(y_test,y_pred)
	print metrics.classification_report(y_test, y_pred)



#apply classification on original vectors
print "\nClassifying Original Vectors"
print "\nVectors Shape(Doc X Terms): %.1f x %.1f" % (vectors.shape[0], vectors.shape[1])
X1 = vectors.toarray()
y1 = d_labels
X_train1, X_test1, y_train1, y_test1 = cross_validation.train_test_split(X1, y1, test_size=0.4, random_state=0)
Classification(X_train1, X_test1, y_train1, y_test1)


#These X2 is the doc-term matrix and y2 is corresponding class labels (target values)
X2 = new_vectors.toarray()
y2 = new_dlabel


#select features using chi square technique and apply classification
#number of terms selected are same in chi squared and proposed selection method for each parameters
#but the top terms selected vary in both cases
print "\nClassifying Chi Vectors"
ch2 = SelectKBest(chi2, k=(new_vectors.shape[1]))
X_train1 = ch2.fit_transform(X_train1, y_train1)
X_test1 = ch2.transform(X_test1)
print "\nVectors Shape(Doc X Terms): %.1f x %.1f" % (len(X_train1)+len(y_train1), len(X_train1[0]))
Classification(X_train1, X_test1, y_train1, y_test1)



#apply classification on vectors obtained after feature selection
print "\nClassifying Feature Reduced Vectors"
print "\nNew Vectors Shape(Doc X Terms): %.1f x %.1f" % (new_vectors.shape[0], new_vectors.shape[1])

X_train2, X_test2, y_train2, y_test2 = cross_validation.train_test_split(X2, y2, test_size=0.4, random_state=0)
Classification(X_train2, X_test2, y_train2, y_test2)

#Also find the ACCURACY VALUES

#THE END
