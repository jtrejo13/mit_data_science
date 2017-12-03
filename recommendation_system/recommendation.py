import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split



col_names = ['user_id', 'item_id', 'rating', 'timestamp']
data = pd.read_table('u.data', names=col_names)
data = data.drop('timestamp', 1)
data.info()

plt.hist(data['rating'])
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Rating Histogram')
plt.show()

number_ratings = len(data)
number_movies  = len(np.unique(data.item_id))
number_users   = len(np.unique(data.user_id))

sparcity = float(number_ratings) / (number_movies * number_users) * 100.0

print(sparcity)

# -----------------
# splitting the data
# -----------------

train, test = train_test_split(data, test_size=0.3)

print(len(train))
print(len(test))




