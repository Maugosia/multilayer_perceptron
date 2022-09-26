from sklearn import datasets, preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def get_house_prediction_data():
    df = pd.read_csv("house_price_data.csv")
    data = df.values.tolist()
    y = [row[-1] for row in data]
    x = [row[0:-1] for row in data]
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    return x, y


def read_mnist_data():
    mnist_dataset = datasets.load_digits(n_class=10)
    x, y = mnist_dataset.data, mnist_dataset.target
    return x, y


def get_train_test_validation(percent_test, percent_val):
    min_max_scaler = preprocessing.MinMaxScaler()
    one_hot_encoder = preprocessing.OneHotEncoder(sparse=False)

    x, y = read_mnist_data()
    x = x[0:400]
    y = y[0:400]
    columns_names = []
    for i in range(0, np.shape(x)[1]):
        columns_names.append(str(i))

    x_names = np.copy(columns_names)
    columns_names.append("class")

    df = pd.DataFrame(data=np.concatenate((x, y.reshape((len(y), 1))), axis=1), columns=columns_names)
    print("class proportions = ")
    print(df["class"].value_counts(normalize=True))

    train_validation, test = train_test_split(df, test_size=percent_test, shuffle=True, stratify=df["class"])
    train, validation = train_test_split(train_validation, test_size=percent_val,
                                         shuffle=True, stratify=train_validation["class"])

    train_x = train_validation[x_names].to_numpy()
    test_x = test[x_names].to_numpy()
    validation_x = validation[x_names].to_numpy()

    train_y = train_validation["class"].to_numpy()
    test_y = test["class"].to_numpy()
    validation_y = validation["class"].to_numpy()

    min_max_scaler.fit(x)
    train_x = min_max_scaler.transform(train_x)
    test_x = min_max_scaler.transform(test_x)
    validation_x = min_max_scaler.transform(validation_x)

    one_hot_encoder.fit(y.reshape((len(y), 1)))
    test_y = one_hot_encoder.transform(test_y.reshape((len(test_y), 1)))
    train_y = one_hot_encoder.transform(train_y.reshape((len(train_y), 1)))
    validation_y = one_hot_encoder.transform(validation_y.reshape((len(validation_y), 1)))
    
    return train_x, train_y, test_x, test_y, validation_x, validation_y
