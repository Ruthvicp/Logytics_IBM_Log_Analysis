from unittest.mock import inplace
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

df = pd.read_excel(r"C:\Users\Karthik Katta\Desktop\DataFormed.xlsx")
df.convert_objects(convert_numeric= True)
df.fillna(0, inplace = True)

data = []

for i in range(len(df["ERROR_MESSAGE"])):
    #print(df["ERROR_MESSAGE"][i].split(' '))
    data.append(df["ERROR_MESSAGE"][i])

print(data[10])

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data)

print(vectorizer.get_feature_names())
print(X.shape)

clf = KMeans(n_clusters= 8, random_state= 1)
clf.fit(X)
y_kmeans = clf.predict(X)

print(clf.labels_)


print("Top terms per cluster:")
order_centroids = clf.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(5):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :5]:
        print(' %s' % terms[ind]),

pca = TruncatedSVD(n_components=2)
pca.fit(X)
X1=pca.fit_transform(X)
# plt.scatter(X1[:, 0], X1[:, 1])
# centers = clf.cluster_centers_
# #plt.scatter(X1[:, 0], X1[:, 1], c='blue', s=200, alpha=0.5);
# plt.show()

plt.scatter(X1[:, 0], X1[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = clf.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()
