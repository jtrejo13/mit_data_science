# -------
# Imports
# -------

import pandas as pd
from graphlab import SFrame
from graphlab import item_similarity_recommender as sr


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

# ----------------------
# Similarity Recommender
# ----------------------

recommender = sr.create(train, target='rating')
rmse = recommender.evaluate_rmse(test, target='rating')

print(rmse['rmse_overall'])

pres_recall = recommender.evaluate_precision_recall(test)
print(pres_recall)
