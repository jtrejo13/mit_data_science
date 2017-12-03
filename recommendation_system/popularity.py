# -------
# Imports
# -------

import pandas as pd
from graphlab import SFrame
from graphlab import popularity_recommender as pr


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
print(len(train))
print(len(test))

# ----------------------
# Popularity Recommender
# ----------------------

recommender = pr.create(train, target='rating')
eval = recommender.evaluate(test)  # ('\nOverall RMSE: ', 1.0262364153594763)
