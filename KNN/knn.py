import numpy as np


class MyKNearestNeighborsRegressor:
    def __init__(self, count_nearest_neighbors=5):
        self.count_nearest_neighbors = count_nearest_neighbors

    def fit(self, X_train, y_train):
        self.X_train, self.y_train = np.array(X_train), np.array(y_train)
        
    def predict(self, X):
        y_predict = []

        for x_val in X:
            neighbors_ids = self.get_neighbors_ids(self.X_train, x_val, self.count_nearest_neighbors)
            nearest_values = [self.y_train[id] for id in neighbors_ids] 

            y_predict.append(np.mean(nearest_values))

        return y_predict
    
    def dist_between_vecs(self, vec1, vec2):
        # distance = 0
        # for i in range(len(vec1)):
        #     distance += (vec1[i] - vec2[i])**2

        # return distance**0.5
        return np.linalg.norm(vec1, vec2)
    
    def get_neighbors_ids(self, train_x, target_row, k):
        dist_list = []
        for id, row in enumerate(train_x):
            dist = self.dist_between_vecs(row, target_row)
            dist_list.append([id, dist])

        dist_list.sort(key=lambda x: x[1])

        return list(map(lambda x: x[0], dist_list[:k]))

