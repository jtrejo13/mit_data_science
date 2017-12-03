# -------
# Imports
# -------

import pandas as pd
import numpy as np
from graphlab import SFrame
from graphlab import factorization_recommender as fr
from graphlab import evaluation

# -----------
# Data Import
# -----------


col_names = ['user_id', 'item_id', 'rating', 'timestamp']
data = pd.read_table('u.data', names=col_names)
data = data.drop('timestamp', 1)

# ----------
# Data Split
# ----------

sf = SFrame(data=data, format='auto')
train, test = sf.random_split(.7)
train, validation = train.random_split(.75)
print(len(train))          # 53025
print(len(validation))     # 17169
print(len(test))           # 29806

# --------------------------
# Collaborative Recommender
# --------------------------

# regularization parameter range
reg_parameters = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1]
print(reg_parameters)

ideal_param = 0
min_rmse = np.inf

for param in reg_parameters:
    recom = fr.create(train, target='rating', linear_regularization=param, verbose=False)
    rmse = recom.evaluate_rmse(validation, target='rating')
    if rmse['rmse_overall'] < min_rmse:
        ideal_param = param
        min_rmse = rmse['rmse_overall']


print(ideal_param)   # 0.001
print(min_rmse)      # 0.943


# -----------
#  Model
# -----------

ideal_recommender = fr.create(test, target='rating', linear_regularization=ideal_param, verbose=True)

# Computing final objective value and training RMSE.
#        Final objective value: 1.03222
#        Final training RMSE: 0.933035


print(ideal_recommender.recommend())
