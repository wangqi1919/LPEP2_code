from model_functions import DataPreprocess
from model_functions import ModelTrain
import numpy as np
import os
import sklearn
import statistics


if __name__ == "__main__":
    data_version = 'shortdata_onehot'
    data_path = 'data/' + data_version + '.csv'
    validation = True
    feature_name = ['maxdepth_m', 'IWS_lk_ratio', 'iws_nlcd2006_for','lake_sdf', 'lakeconn_v2_1', 'lakeconn_v2_2',
                    'iws_sdf', 'iws_streamdensity_streams_density_mperha', 'elevation_m','wlconnections_allwetlands_shoreline%',
                    'roaddensity_mperha','baseflow_mean','runoff_mean','totalNdep_19902010_diff','hu4_nlcd2006_agr','totalNdep_1990_mean',
                    'prism_ppt_mean','prism_tmean_mean', 'secchi']
    conditional = 0 # if 1, conditioanl only on secchi; if 0, no conditioanl; if 2, conditional on all response
    val_times = 5
    splitting_string = ['random25', 'random75', 'hu4_ago', 'hu4_strat', 'hu4_random', 'cluster_strat75', 'cluster_random50_holdout']
    responses_all = ['TP', 'TN', 'chla', 'secchi']
    conditioanl_string = ['notconditional', 'conditionalsecchi', 'conditioanl_all']
    result_dir = 'Results_scaled/TrainedModels_' + conditioanl_string[conditional] + '/'+'validation' + str(validation) + '/' + data_version + '/'
    for response in [1,2,3,4]:
        for splitting_method in [1, 2, 3, 4, 5, 6, 7]:
            data = DataPreprocess.Data(data_path, splitting_method, response, conditional)
            data.load_data()
            Xtrain, Xtest, Xtest_nonzscore, ytrain, ytest, zonetrain, zonetest, type_test, position_test, lakeid_test, lakeid_train = data.preprocess()
            model_name = 'randomforest'
            parameters = [[50, 50], [100, 100],  [100, 50], [200, 200]]
            model_dir = result_dir + responses_all[response-1] + '/' + splitting_string[splitting_method-1] + '/' + model_name + '/'
            model_path = model_dir + 'model'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model = ModelTrain.Model(model_name, parameters, Xtrain, Xtest, ytrain, ytest, zonetrain, zonetest, validation, model_path, val_times)
            RMSE_tr, RMSE_te, RMSE_val, ypred_test, ypred_train = model.TrainModel()
            r_squared = sklearn.metrics.r2_score(ytest, ypred_test)
            abs_r_squared = sklearn.metrics.r2_score(np.exp(ytest) - 0.1, np.exp(ypred_test) - 0.1)
            abs_relative_error = np.divide(abs(np.exp(ypred_test) - np.exp(ytest)), abs(np.exp(ytest)-0.1))
            median_error_relative = statistics.median(abs_relative_error)

            print responses_all[response-1]
            print splitting_string[splitting_method-1]
            print RMSE_te
            print r_squared
            print abs_relative_error
            print median_error_relative














