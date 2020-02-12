
# coding: utf-8

# In[1]:


import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np 
from scipy.signal import welch
from detect_peaks import detect_peaks
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import entropy #seems non beneficial
from scipy.signal import hilbert


# In[2]:


#Import Machine Learning Libraries
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# In[3]:


#Get Parameters from GUI
#Hardcoded here
#needs updating
n = 2000 / 60
N = 16
Bd = 0.331*254
Pd = 2.815*254
phi = 15.17 * np.pi / 180
SampleFrequency = 20000
FileOfInterest = '2003.10.22.12.06.24'
HomeDirectory = os.getcwd()
os.chdir(HomeDirectory)
directory = os.listdir(HomeDirectory)
TrainingDataFile = "DELETE.csv"


# In[4]:


#Get Training Data
for file in directory:
    if file == TrainingDataFile:
        dataset = pd.read_csv(file,header = 0,index_col = 0)

X = dataset.values[:,1:(dataset.shape[1]-1)]
Y = dataset.values[:,0]
validation_size = 0.20
seed = 6
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)   


# In[5]:


#Train Model
clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train, Y_train)
print("Accuracy on training set is : {}".format(clf.score(X_train, Y_train)))
print("Accuracy on test set is : {}".format(clf.score(X_test, Y_test)))
Y_test_pred = clf.predict(X_test)
print(classification_report(Y_test, Y_test_pred))


# In[6]:


data = pd.read_table(FileOfInterest,header = None)
data.columns = ['b1x','b1y','b2x','b2y','b3x','b3y','b4x','b4y']
b1x = np.transpose(data.values[:,0])


# In[7]:


#Compute need info using GUI inputs and data information
NumberOfSamples = len(b1x)
dt = 1/SampleFrequency
Tmax = dt*NumberOfSamples
t = np.arange(0,Tmax,dt) #same as x*dt
print(Tmax)


# In[13]:


from Functions import getCompleteDataFrame
from Functions import getGraphs
TEST = getCompleteDataFrame(b1x,Tmax,NumberOfSamples,SampleFrequency,n,N,Bd,Pd,phi,FileOfInterest,1)
TEST = TEST.values[:,1:(TEST.shape[1]-1)]
OUTCOME = clf.predict(TEST)
print(OUTCOME)
figs = getGraphs(b1x,Tmax,NumberOfSamples,SampleFrequency,n,N,Bd,Pd,phi,file,1)

