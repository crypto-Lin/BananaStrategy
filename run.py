from core.model import *
from core.evaluation import *

import json
import xgboost as xgb
import os
import datetime as dt
from sklearn.model_selection import GridSearchCV

# train_file_path = './dataset/A_stock_daily/train.csv'
# test_file_path = './dataset/A_stock_daily/test.csv'
train_file_path = './dataset/A_stock_5d/train.csv'
test_file_path = './dataset/A_stock_5d/test.csv'
configs = json.load(open('config.json', 'r'))


def main():
    if not os.path.exists(configs['model_params']['save_dir']): os.makedirs(configs['model_params']['save_dir'])
    save_dir = configs['model_params']['save_dir']
    save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), 'xgb'))

    df_train = pd.read_csv(train_file_path)
    x_train = df_train[configs['factor_feature_extract']]
    y_train = df_train[configs['factor_feature_extract'][-1]]
    print('训练目标值分布：')
    print(y_train.value_counts())

    df_test = pd.read_csv(test_file_path)
    x_test = df_test[configs['factor_feature_extract']]
    y_test = df_test[configs['factor_feature_extract'][-1]]

    train_params = configs['model_params']['train_params']
    clf = xgb.XGBClassifier(**train_params)
    clf.fit(x_train, y_train,
        eval_set=[(x_train, y_train), (x_test, y_test)],
        eval_metric='logloss', # 'auc'
        early_stopping_rounds=5,
        verbose=True, callbacks = [xgb.callback.EarlyStopping(rounds=5,metric_name='auc',save_best=True),
                                   xgb.callback.TrainingCheckPoint(directory=save_fname,name='xbg_binary_classifier')])

    evaluate_result = clf.evals_result()
    print(confusion_matrix_model(clf, x_test, y_test))

    # check the fit params
    print('Get underlying booster of the model:{}'.format(clf.get_booster()))
    print('Gets the number of xgboost boosting rounds:{}'.format(clf.get_num_boosting_rounds()))
    print('Feature importance property:{}'.format(clf.feature_importances_))

    # plotting xgboost
    clf.plot_importance()
    clf.plot_tree()

    # cv_params = configs['model_params']['cv_params']
    cv_params = {}
    # model = xgb.XGBClassifier(**train_params)
    # optimized_gbm = GridSearchCV(estimator=model, param_grid=cv_params, scoring='roc_auc', cv=5, verbose=1, n_jobs=4)
    # optimized_gbm.fit(x_test, y_test)
    # evaluate_result = optimized_gbm.cv_results_
    print('每轮迭代运行结果:{0}'.format(evaluate_result))
    #print('参数的最佳取值：{0}'.format(optimized_gbm.best_params_))
    #print('最佳模型得分:{0}'.format(optimized_gbm.best_score_))



    #importance_of_features(model.fit(x_train, y_train))





if __name__ == '__main__':
    main()
