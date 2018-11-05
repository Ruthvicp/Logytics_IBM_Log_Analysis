import io

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

path = "C:\\Users\\ruthv\\PycharmProjects\\LOGytics\\data\\latest\\1.txt"
with io.open(path, encoding='utf-8') as f:
    documents = f.read().lower()
    documents = documents.split('\n')

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

true_k = 2
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

print("\n")
print("Prediction")

Y = vectorizer.transform(["Network listener connection."])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["received connection."])
prediction = model.predict(Y)
print(prediction)