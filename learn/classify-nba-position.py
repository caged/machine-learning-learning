# Given a set of player statistics, can we tell what position they play?

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("data/nba-players-basic.csv")

positions = data.pos.unique().tolist()

def pindex(row):
    return positions.index(row)

data['pindex'] = data['pos'].apply(pindex)

print(data)

training_data = data[data.season == "2014-15"]
test_data = data[data.season == "2015-16"]

trX = training_data.pos
trY = training_data.ast

teX = test_data.pos.tolist()
teY = test_data.ast.tolist()

knn = KNeighborsClassifier()
knn.fit([trX], [trY])
