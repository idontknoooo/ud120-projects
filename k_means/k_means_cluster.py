#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""



import os
import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cluster import KMeans



def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 

# Quiz3: Clustering Features
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = 'total_payments'
poi  = "poi"
#features_list = [poi, feature_1, feature_2, feature_3]
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )

# Quiz6: Stock Option Range
a1 = max(i[1] for i in finance_features)
b1 = min(i[1] for i in finance_features if i[1] > 0)
# Quiz7: Salary Range
a2 = max(i[0] for i in finance_features)
b2 = min(i[0] for i in finance_features if i[0] > 0)
# Quiz8: Check if the 'cluster1.pdf' is the same as it shows in quiz. If not, find the different point. The reason they are changing is because, this clustering is not obey scale-invariance.

### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)

# Quiz5: Clustering with 3 features
#for f1, f2, _ in finance_features:
#    plt.scatter( f1, f2 )
#    plt.scatter( f1, _ )
#    plt.scatter( f2, _ )
#plt.show()

for f1, f2 in finance_features:
    plt.scatter(f1, f2)
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred

# Quiz4: Deploying Clustering 
kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
pred = KMeans.predict(kmeans, data) 


try:
    os.remove("clusters.pdf")
except:
    print "No clusters.pdf" 
### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:

    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"
