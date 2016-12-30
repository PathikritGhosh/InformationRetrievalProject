__author__ = "Pathikrit Ghosh"
__version__ = "1.0.1"
__python_version___ = "2.7"

import shutil,sys,math,time,os,string
import nltk
import heapq
import itertools
import numpy as np
import cPickle as pickle
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
from sklearn import metrics
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
import warnings
warnings.filterwarnings("ignore")


cwd = os.getcwd()

# '''
# Place the webkb dataset (data files) on creating a folder by name "WebKB".
# Create another folder "WebKB_Dataset" and 4 sub folders by name "course", "faculty", "project, "student".
# '''

dataset_name = "WebKB"
new_class_dir = cwd + '/' + dataset_name + '_Dataset'
list_classes = os.listdir(new_class_dir+'/')
no_of_classes = len(list_classes)
print "New Class dirctory info:"
print "new_class_dir =", new_class_dir
print "list_classes =", list_classes
print "no_of_classes =", no_of_classes,"\n"

#Cleaning up the new directory
for class_index in xrange(no_of_classes):
	dir_path = new_class_dir + '/' + list_classes[class_index]
	fileList = os.listdir(dir_path)
	for fileName in fileList:
		os.remove(dir_path+"/"+fileName)

# first argument passed to the program is top top_num % of features from feature selection by variance
# second argument passed to the program is top top_num2 % of features from feature selection by idf values
# 3rd argument passed to the program is the number of files
# that will be copied from the "WebKB" to corresponding folders ("course", "faculty", "project", "student").
top_num = int(sys.argv[1])
doc_top = 95
top_num2 = int(sys.argv[2])
num2 = top_num*top_num2
no_of_docs = [int(sys.argv[3]) for i in xrange(no_of_classes)]

orig_class_dir = cwd + '/' + dataset_name
list_classes = os.listdir(orig_class_dir+'/')
no_of_classes = len(list_classes)
print "Old Class dirctory info:"
print "new_class_dir =", orig_class_dir
print "list_classes =", list_classes
print "no_of_classes =", no_of_classes,"\n"

for class_index in xrange(no_of_classes):
	for j in xrange(0,no_of_docs[class_index]):
		dest = new_class_dir + '/' + list_classes[class_index]
		num = "%d" % (j) #
		name = "/"+dataset_name+"/"+list_classes[class_index]+'/'+list_classes[class_index]+'_'+str(num)+".txt" #
		source = cwd + name
		shutil.copy2(source, dest)

#Load The files/dataset
load_path = cwd + "/"+dataset_name+"_Dataset"
dataset=load_files(load_path, description=None, categories=None, load_content=True, shuffle=False, encoding=None, decode_error='strict', random_state=0)
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

#gives only tf matrix. Later we get idf by dividing tf-idf value by tf value
idf_vec = TfidfTransformer(use_idf=False,smooth_idf=False).fit(data)
idf_vectors = idf_vec.transform(data)

# find centroid for documents of a class
def document_reduction(vectors,a,begin):
	'''
		this method takes the document vector and the starting position of the class in the vector and also the size of the class.
		then it finds the avg of the document vectors i.e. the centroid vector for the class.
		it computes the cosine similarity with each of the 'a' documents of the class and selects top (doc_top) percentage of documents from that class
		and returns sorted list of top documents of the class based on cosine similarity with the centroid		 
	'''
	centroid_doc = vectors[begin]
	for i in range((begin+1),a):
		centroid_doc+=vectors[i]

	centroid_doc /= a

	#find cosine similarity of class a centroid with all document vectors
	similarity = cosine_similarity(centroid_doc,vectors[begin:(begin+a)])
	top_documents =  heapq.nlargest(int(a*0.01*doc_top), zip(similarity[0], itertools.count(begin)))
	documents_to_keep = [item[1] for item in top_documents]
	return  sorted(documents_to_keep)

#done to remove few outliers:hard coded to 5%
class_doc = []
r1 = 0
r2 = no_of_docs[0]
for class_index in xrange(no_of_classes):
	r2 = no_of_docs[class_index]
	class_doc.append(document_reduction(vectors, r2, r1))
	r1 += r2

cl_voc = []
for item in data:
	cl_voc.append([i for i, j in enumerate(item) if j != 0])

#reject terms of documents that is considered an outlier
class_voc = []
for item in class_doc:
	temp = [cl_voc[x] for x in item]
	class_voc.append(list(set(list(itertools.chain.from_iterable(temp)))))

#get the term and corresonding idf values for each term
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
	return sel_indices2

#Sel_f_indices has all the vocabulary ranked (based on technique)
#taking the top 'M' features and re-building vectors i.e.: doc-term tfidf matrix
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
	
new_vectors = csr_matrix(s)
new_vectors = new_vectors.transpose()

vector2 = vector2.transpose()

def Classification(X_train, X_test, y_train, y_test):
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

print "\nClassifying Original Vectors"
print "\nVectors Shape(Doc X Terms): %.1f x %.1f" % (vectors.shape[0], vectors.shape[1])
X1 = vectors.toarray()
y1 = d_labels
X_train1, X_test1, y_train1, y_test1 = cross_validation.train_test_split(X1, y1, test_size=0.4, random_state=0)
Classification(X_train1, X_test1, y_train1, y_test1)

X2 = new_vectors.toarray()
y2 = new_dlabel

#number of terms selected are same in chi squared and proposed selection method for each parameters
#but the top terms selected vary in both cases
print "\nClassifying Chi Vectors"
ch2 = SelectKBest(chi2, k=(new_vectors.shape[1]))
X_train1 = ch2.fit_transform(X_train1, y_train1)
X_test1 = ch2.transform(X_test1)
print "\nVectors Shape(Doc X Terms): %.1f x %.1f" % (len(X_train1)+len(y_train1), len(X_train1[0]))
Classification(X_train1, X_test1, y_train1, y_test1)

#These X2 is the doc-term matrix and y2 is corresponding class labels (target values)
print "\nClassifying Feature Reduced Vectors"
print "\nNew Vectors Shape(Doc X Terms): %.1f x %.1f" % (new_vectors.shape[0], new_vectors.shape[1])

X_train2, X_test2, y_train2, y_test2 = cross_validation.train_test_split(X2, y2, test_size=0.4, random_state=0)
Classification(X_train2, X_test2, y_train2, y_test2)

#Also find the ACCURACY VALUES

#REPLACING THE FUNCTION "TFIDFMODIFIED" WITH ANYOYTHER FEATURE SELECTION TECHNIQUE WILL PRODUCE CORRESPONDING RESULTS.
#THE END
