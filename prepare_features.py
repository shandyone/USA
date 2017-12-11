from sklearn import preprocessing
import numpy as np

'''
Sale_data,
Sale_price,
Bedroom,
Bathroom,
Average,
Park,
Floor,
Rank,
Architecture,
Basement,
Build_year,
Repair_year,
Longitude,
Latitude
'''

def feature_list(record):
    Sale_data = record['Sale_data']
    Bedroom = record['Bedroom']
    Bathroom = record['Bathroom']
    Average = record['Average']
    Park = record['Park']
    Floor = record['Floor']
    Rank = record['Rank']
    Architecture = record['Architecture']
    Basement = record['Basement']
    Build_year = record['Build_year']
    Repair_year = record['Repair_year']
    Longitude = record['Longitude']
    Latitude = record['Latitude']
    return [
        Sale_data,
        Bedroom,
        Bathroom,
        Average,
        Park,
        Floor,
        Rank,
        Architecture,
        Basement,
        Build_year,
        Repair_year,
        Longitude,
        Latitude
    ]

train_data_x = []
train_data_y = []

for record in train_data:
    if record['Sales'] != '0' and record['Open'] != '':
        fl = feature_list(record)
        train_data_x.append(fl)
        train_data_y.append(int(record['Sale_price']))


#try to
full_X = train_data_x
full_X = np.array(full_X)
train_data_X = np.array(train_data_x)
les = []
for i in range(train_data_X.shape[1]):
    le = preprocessing.LabelEncoder()
    le.fit(full_X[:, i])
    les.append(le)
    train_data_X[:, i] = le.transform(train_data_X[:, i])