import numpy as np
from scipy import stats


def take_log(y):
    y = y + 0.1
    y = np.log(y)
    return y


class Data:
    def __init__(self, data_path, splitting_method, response, conditional):
        self.data_path = data_path
        self.splitting_method = splitting_method
        self.response = response
        self.X = None
        self.y = None
        self.splitting = None
        self.conditional = conditional
        self.zone = None
        self.position = None
        self.type = None
        self.lakeid = None

    def load_data(self):
        # current version only fit shortdata
        data = np.genfromtxt(self.data_path, dtype=float, delimiter=',',skip_header=1)
        self.splitting = data[:, self.splitting_method + 3]
        self.zone = data[:, 1]
        self.position = data[:, 2:4]
        self.type = np.isnan(np.expand_dims(data[:, 14], 1))
        self.lakeid = data[:, 0]
        if self.conditional == 2:
            self.X = data[:, 11:]
            self.y = self.X[:, self.response-1]
            self.X = np.delete(self.X, self.response-1, 1)
        if self.conditional == 0:
            self.X = data[:, 15:]
            self.y = data[:, 11 + self.response - 1]
        if self.conditional == 1:
            self.X = data[:, 15:]
            self.y = data[:, 11 + self.response - 1]
            # add secchi when the target is not secchi for conditional models
            if self.response != 4:
                secchi = np.expand_dims(data[:, 14], 1)
                self.X = np.concatenate((self.X, secchi), axis=1)

    def preprocess(self):
        # remove data with response == NaN
        ID_not_NaN = ~np.isnan(self.y)
        self.y = self.y[ID_not_NaN]
        self.X = self.X[ID_not_NaN, :]
        self.zone = self.zone[ID_not_NaN]
        self.lakeid = self.lakeid[ID_not_NaN]
        self.type = self.type[ID_not_NaN]
        self.position = self.position[ID_not_NaN]
        # using mean to replace NaN in X
        col_mean = np.nanmean(self.X, axis=0)
        inds = np.where(np.isnan(self.X))
        self.X[inds] = np.take(col_mean, inds[1])
        # splittiing trainning, testing
        splitting = self.splitting[ID_not_NaN]
        tr_id = splitting == 0
        te_id = splitting == 1
        Xtest_nonzscore = self.X[te_id, :]
        self.X = stats.zscore(self.X)
        # take log to all y
        self.y = take_log(self.y)
        Xtrain = self.X[tr_id, :]
        Xtest = self.X[te_id, :]
        ytrain = self.y[tr_id]
        ytest = self.y[te_id]
        zonetrain = self.zone[tr_id]
        zonetest = self.zone[te_id]
        type_test = self.type[te_id]
        position_test = self.position[te_id]
        lakeid_test = self.lakeid[te_id]
        lakeid_train = self.lakeid[tr_id]
        return Xtrain, Xtest, Xtest_nonzscore, ytrain, ytest, zonetrain, zonetest, type_test, position_test, lakeid_test, lakeid_train
