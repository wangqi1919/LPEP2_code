import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn import neighbors
from sklearn.linear_model import Ridge
import os.path
import pickle


class Model:
    def __init__(self, model_name, parameters, Xtrain, Xtest, ytrain, ytest, zonetrain, zonetest, validation, final_model_path, val_times=5):
        self.model_name = model_name
        self.parameters = parameters
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.ytrain = ytrain
        self.ytest = ytest
        self.zonetrain = zonetrain
        self.zonetest = zonetest
        self.validation = validation
        self.val_times = val_times
        self.final_model_path = final_model_path


    def BuildModel(self, para, Xtr, ytr):
        model = RandomForestRegressor(max_depth=para[0], n_estimators=para[1])
        model.fit(Xtr, ytr)
        return model

    def TrainModel(self):
        if not self.validation:
            if not os.path.isfile(self.final_model_path + '.pkl'):
                validation_rmse_all = []
                for para in self.parameters:
                    model_tune = self.BuildModel(para, self.Xtrain, self.ytrain)
                    validation_pred = model_tune.predict(self.Xtest)
                    validation_rmse_all.append(sqrt(mean_squared_error(validation_pred, self.ytest)))
                best_para = np.argmin(validation_rmse_all)
                RMSE_val = np.min(validation_rmse_all)
                model_final = self.BuildModel(self.parameters[best_para], self.Xtrain, self.ytrain)
                with open(self.final_model_path  + '.pkl', 'wb') as output:
                    pickle.dump({'model_final': model_final, 'RMSE_val': RMSE_val, 'best_para': self.parameters[best_para]}, output, pickle.HIGHEST_PROTOCOL)

        if self.validation:
            if not os.path.isfile(self.final_model_path + '.pkl'):
                validation_rmse_all = []
                n, d = self.Xtrain.shape
                for iter in range(self.val_times):
                    print "iter" + str(iter)
                    id = np.arange(n)
                    val_id = id % self.val_times == iter
                    tr_id = id % self.val_times != iter
                    Xval = self.Xtrain[val_id, :]
                    Xtr = self.Xtrain[tr_id, :]
                    yval = self.ytrain[val_id]
                    ytr = self.ytrain[tr_id]
                    validation_rmse_oneiter = []
                    for para in self.parameters:
                        model_tune = self.BuildModel(para, Xtr, ytr)
                        validation_pred = model_tune.predict(Xval)
                        validation_rmse_oneiter.append(sqrt(mean_squared_error(validation_pred, yval)))
                    validation_rmse_all.append(validation_rmse_oneiter)
                # find the best parameter and trian the model on the whole train dataset
                validation_rmse_mean = np.mean(validation_rmse_all, axis=0)
                RMSE_val = np.min(validation_rmse_mean)
                best_para = np.argmin(validation_rmse_mean)
                model_final = self.BuildModel(self.parameters[best_para], self.Xtrain, self.ytrain)
                with open(self.final_model_path + '.pkl', 'wb') as output:
                    pickle.dump({'model_final': model_final, 'RMSE_val': RMSE_val, 'best_para': self.parameters[best_para]}, output,
                                pickle.HIGHEST_PROTOCOL)

        with open(self.final_model_path + '.pkl', 'rb') as input:
            saved_model = pickle.load(input)
            model_final = saved_model['model_final']
            RMSE_val = saved_model['RMSE_val']
            ypred_test = model_final.predict(self.Xtest)
            RMSE_te = sqrt(mean_squared_error(ypred_test, self.ytest))
            ypred_train = model_final.predict(self.Xtrain)
            RMSE_tr = sqrt(mean_squared_error(ypred_train, self.ytrain))
        return RMSE_tr, RMSE_te, RMSE_val, ypred_test, ypred_train


