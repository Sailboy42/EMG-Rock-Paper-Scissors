import numpy as np
import matplotlib.pyplot as plt #imported for plotting
import scipy.io #used for computing values
import pickle #importing to save
import ChosenFeatures as ChosenFeatures # I created this list of 6 chosen featur es 
import Functions as Functions # holds various functions I mde to be used here
from sklearn.metrics import accuracy_score # using to get accuracy of data
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from ChosenFeatures import extract_features
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from scipy.signal import butter, filtfilt
from PreProcess import preprocess_data #import function I made to preprocess data

def highpass(hpdata, cutoff = 20, hpfs = 1000):
    b, a = butter(1, cutoff, "hp", fs=hpfs, analog=False, output="ba")
    return filtfilt(b, a, hpdata, axis=0, padlen=2)

class RunPythonModel:
    #load model
    def __init__(self, modelPath):
        with open(modelPath, 'rb') as file:
            self.model = pickle.load(file)
        


    def get_rps(self,data):
        """
        Function to take in data and return the rps. You can
        place this function wherever you want, but add code here that
        takes in the data and returns rock, paper, or scissors
        after putting the data through your model.
        """

        np_data = data[:, 1:-1]
        #pre process
        FilteredData = highpass(np_data) #this returns filtered data
        # data file is an array in the form of (time,rock,paper,scissors)

        #extract features using chosen features.py
        #preprocessed returns data in format of (time,1,2,3)

        ExtractedFeatures = ChosenFeatures.extract_features(FilteredData.T) 

        #reshape by 1,-1 to let sy py do classification 
        #call self.model.predict with x test and add 1

        return self.model.predict(ExtractedFeatures.reshape([1, -1]))[0]+1 # retunrs an array with a 0,1, or 2









