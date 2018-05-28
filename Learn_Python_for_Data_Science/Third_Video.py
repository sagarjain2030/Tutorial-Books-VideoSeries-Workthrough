import  numpy as np
import scipy
import scipy.sparse as sp
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#Challenge of writing movielens function:
#Considering Movie lens dataset

#Please download MovieLens 100K Dataset from https://grouplens.org/datasets/movielens/
#Extract it to current path/ml-100k
def parseItemMetadata(genres_raw,num_items,item_metadata_raw):

    genres = []

    for line in genres_raw:
        if line:
            genre, gid = line.split('|')
            genres.append('genre:{}'.format(genre))

    id_feature_labels = np.empty(num_items, dtype=np.object)
    genre_feature_labels = np.array(genres)

    id_features = sp.identity(num_items,
                              format='csr',
                              dtype=np.float32)
    genre_features = sp.lil_matrix((num_items, len(genres)),
                                   dtype=np.float32)

    for line in item_metadata_raw:

        if not line:
            continue

        splt = line.split('|')

        # Zero-based indexing
        iid = int(splt[0]) - 1
        title = splt[1]

        id_feature_labels[iid] = title

        item_genres = [idx for idx, val in
                       enumerate(splt[5:])
                       if int(val) > 0]

        for gid in item_genres:
            genre_features[iid, gid] = 1.0

    return (id_features, id_feature_labels,
            genre_features.tocsr(), genre_feature_labels)


def parseData(data):
    for line in data:
        if not line:
            continue

        uid,iid,rating,timestamp = [x for x in line.split('\t')]
        #make it zero based index
        yield int(uid)-1,int(iid)-1,int(rating),timestamp

def getDimension(trainData, testData):
    uids = set()
    iids = set()

    for uid,iid,_,_ in iter(trainData):
        uids.add(uid)
        iids.add(iid)

    for uid,iid,_,_ in iter(testData):
        uids.add(uid)
        iids.add(iid)

    rows = max(uids) + 1
    column = max(iids) + 1

    return rows,column

def buildMatrixConfig(rows,columns, data):
    mat = sp.lil_matrix((rows,columns),dtype=np.int32)
    for uid, iid, rating, _ in data:
        mat[uid, iid] = rating

    return mat.tocoo()

def fetchMovies():

    #Fetch data for train,test,item_metadata & genre
    with open('ml-100k/ua.base','r') as file1:
        train_raw = file1.read().split('\n')
    with open('ml-100k/ua.test','r') as file1:
        test_raw = file1.read().split('\n')
    with open('ml-100k/u.item','r') as file1:
        item_metadata_raw = file1.read().split('\n')
    with open('ml-100k/u.genre','r') as file1:
        genres_raw = file1.read().split('\n')

    trainingData = parseData(train_raw)
    testingData = parseData(test_raw)

    num_user, num_items = getDimension(trainingData,testingData)

    train = buildMatrixConfig(num_user,num_items,trainingData)
    test = buildMatrixConfig(num_user,num_items,testingData)

    (id_features, id_feature_labels,genre_features,genre_feature_labels) = parseItemMetadata(genres_raw,num_items,item_metadata_raw)

    features = sp.hstack([id_features, genre_features]).tocsr()
    feature_labels = np.concatenate((id_feature_labels,
                                         genre_feature_labels))

    data = {'train': train,
            'test': test,
            'item_features': features,
            'item_feature_labels': feature_labels,
            'item_labels': id_feature_labels}

    return data

#fetch data from movielens datasetwithout any ratings conditions

d = fetchMovies()
data = fetch_movielens(min_rating=4.0)

print(repr(data['train']))
print(repr(data['test']))
print(data['item_features'])
print(data['item_feature_labels'])
print(data['item_labels'])

#Create model
model = LightFM(loss='warp')
#train model
model.fit(data['train'],epochs=30,num_threads=2,verbose=False)

def sample_recommendation(model, data, user_ids):
    # number of users and movies in training data
    n_users, n_items = data['train'].shape

    # generate recommendations for each user we input
    for user_id in user_ids:

        # movies they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        # rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        # print out the results
        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)

def recommendation(model, data, user_ids):
    # number of users and movies in training data
    n_users, n_items = data['train'].shape
    top_items = []
    # generate recommendations for each user we input
    for user_id in user_ids:

        # movies they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        # rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

    return top_items[:10] if len(top_items) > 10 else top_items

sample_recommendation(model, data, [3, 25, 450])

#creating 3 models for 3 different
model1 = LightFM(loss = 'logistic')
model1.fit(data['train'],epochs=30,num_threads=2,verbose=False)
print("Model1 Done")

model2 = LightFM(loss = 'bpr')
model2.fit(data['train'],epochs=30,num_threads=2,verbose=False)
print("Model2 Done")

model3 = LightFM(loss = 'warp')
model3.fit(data['train'],epochs=30,num_threads=2,verbose=False)
print("Model3 Done")

top_items1 = recommendation(model1, data, [3])
top_items2 = recommendation(model2, data, [3])
top_items3 = recommendation(model3, data, [3])


print("top recommendations are")

out  = [x for x in top_items1 if x in top_items2 and x in top_items3]
print(out)
