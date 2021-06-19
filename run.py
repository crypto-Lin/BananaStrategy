from core.lstm_model import *
from core.evaluation import *

import json
import xgboost as xgb
import os
import datetime as dt
from sklearn.model_selection import GridSearchCV

train_file_path = './dataset/A_stock_daily/train.csv'
test_file_path = './dataset/A_stock_daily/test.csv'
# train_file_path = './dataset/A_stock_5d/train.csv'
# test_file_path = './dataset/A_stock_5d/test.csv'
configs = json.load(open('config.json', 'r'))


def main():
    if not os.path.exists(configs['model_params']['save_dir']): os.makedirs(configs['model_params']['save_dir'])
    save_dir = configs['model_params']['save_dir']
    save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), 'xgb'))

    df_train = pd.read_csv(train_file_path)
    # df_train = df_train[(df_train['morning_doji_star']==1) | (df_train['evening_doji_star']==1)]
    x_train = df_train[configs['factor_feature_extract'][:-1]]
    y_train = df_train[configs['factor_feature_extract'][-1]]
    print('训练目标值分布：')
    print(y_train.value_counts())

    df_test = pd.read_csv(test_file_path)
    print(len(df_train), len(df_test))
    # df_test = df_test[(df_test['macd_cross_up_signal']==1) | (df_test['macd_cross_down_signal']==1)]
    x_test = df_test[configs['factor_feature_extract'][:-1]]
    y_test = df_test[configs['factor_feature_extract'][-1]]
    print('测试集目标值分布：')
    print(y_test.value_counts())

    train_params = configs['model_params']['train_params']
    clf = xgb.XGBClassifier(**train_params)
    clf.fit(x_train, y_train,
        eval_set=[(x_train, y_train), (x_test, y_test)],
        eval_metric='logloss', # 'auc'
        early_stopping_rounds=5,
        verbose=True, callbacks = [xgb.callback.EarlyStopping(rounds=5, metric_name='logloss', save_best=True),
                                   xgb.callback.TrainingCheckPoint(directory=save_fname, name='xbg_binary_classifier')])

    evaluate_result = clf.evals_result()
    print(confusion_matrix_model(clf, x_test, y_test))
    df_test['xgb_predict'] = clf.predict(x_test)
    df_test['xgb_predict_proba'] = clf.predict_proba(x_test)[:, 1]
    df_test[['code','open','close','high','low','datetime','xgb_predict','xgb_predict_proba']].to_csv('./dataset/A_stock_daily/test_with_xgb_predict.csv')

    # check the fit params
    print('Get underlying booster of the model:{}'.format(clf.get_booster()))
    print('Gets the number of xgboost boosting rounds:{}'.format(clf.get_num_boosting_rounds()))
    # print('Feature importance property:{}'.format(clf.feature_importances_))
    feature_importance = np.array([clf.feature_importances_])
    print(pd.DataFrame(feature_importance, columns=configs['factor_feature_extract'][:-1]).T)
    # plotting xgboost
    # xgb.plot_importance(clf)
    # xgb.plot_tree(clf)
    plt_roc_curve(x_test, y_test, 'xgb', clf)

    # cv_params = configs['model_params']['cv_params']
    cv_params = {}
    # model = xgb.XGBClassifier(**train_params)
    # optimized_gbm = GridSearchCV(estimator=model, param_grid=cv_params, scoring='roc_auc', cv=5, verbose=1, n_jobs=4)
    # optimized_gbm.fit(x_test, y_test)
    # evaluate_result = optimized_gbm.cv_results_
    print('每轮迭代运行结果:{0}'.format(evaluate_result))
    # print('参数的最佳取值：{0}'.format(optimized_gbm.best_params_))
    # print('最佳模型得分:{0}'.format(optimized_gbm.best_score_))

    # importance_of_features(model.fit(x_train, y_train))


if __name__ == '__main__':
    main()


# xgb params intro:
# Parameters for tree booster
# eta/learning rate
# gamma/min_split_loss : Minimum loss reduction required to make a further partition on a leaf node of the tree.
# The larger gamma is, the more conservative the algorithm will be.
# min_child_weight : Minimum sum of instance weight (hessian) needed in a child.
# If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight,
# then the building process will give up further partitioning.
# The larger min_child_weight is, the more conservative the algorithm will be. range: [0,∞]
# max_delta_step : it might help in logistic regression when class is extremely imbalanced.
# Set it to value of 1-10 might help control the update.
# subsample : Subsample ratio of the training instances. this will prevent overfitting.
# Subsampling will occur once in every boosting iteration.
# tree_method : string [default= auto]
# params = auto, exact, approx, hist, gpu_hist, the larger the data, the righter method to choose
# scale_pos_weight : Control the balance of positive and negative weights, useful for unbalanced classes.
# num_parallel_tree, [default=1] - Number of parallel trees constructed during each iteration.
# This option is used to support boosted random forest.
# Learning task parameters
