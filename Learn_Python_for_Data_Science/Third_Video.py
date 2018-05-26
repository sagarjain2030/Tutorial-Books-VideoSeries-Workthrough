import  numpy as np
import scipy
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetch data from movielens dataset
data = fetch_movielens(min_rating=4.0)

print(repr(data['train']))
print(repr(data['test']))

#Create model
model = LightFM(loss='warp')
#train model
model.fit(data['train'],epochs=30,num_threads=2,verbose=True)

def sample_recommendation(model,data,user_ids):
    n_user,n_movie = data['train'].shape

