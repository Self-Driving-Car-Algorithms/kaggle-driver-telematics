#!/usr/bin/env python
#
# 2014/12/17 - Fabio Gabriel <fmagalhaes@gmail.com>
#
# TODO: Don't use all 200 observations for training data.
# TODO: Implement neural network.

from os import listdir, path
import numpy as np
import pandas as pd
import random
from sklearn import metrics
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNetCV, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

import re
from sys import argv
import gzip

DRIVERS_TABLE = {}
ROOT_DIR = ""
SEED = 0


def sample(toremove, k=3):
    """
    Take all trips from a random sample of k drivers.
    """
    keys = [key for key in DRIVERS_TABLE if key != toremove]
    return random.sample(keys, k)


def open_file(filename):
    item = DRIVERS_TABLE.get(filename)
    if item is not None:
        return item
    else:
        item = pd.read_pickle(path.join(ROOT_DIR,filename))
        DRIVERS_TABLE[filename] = item
        return item


def create_merged_dataset(filename):
    df = open_file(filename)
    n = len(df)
    Y = np.repeat(1, n)
    keys = sample(filename)

    for k in keys:
        other = open_file(k)
        # Just in case we have a file with a different number of lines.
        n_other = len(other)
        df = df.append(other, ignore_index=True)
        Y = np.concatenate((Y, np.repeat(0, n_other)))

    return (df, Y)


def format_submission(predictions):
    output = gzip.open("submission.gz", "w")
    output.write("driver_trip,prob\n")
    for row in predictions.iterrows():
        match = re.match(r"([a-z]+)([0-9]+)", row[1]['driver'], re.I)
        items = match.groups()
        line = "%i_%i,%f\n" % (int(items[1]), int(row[1]['trip']), row[1]['probs'])
        output.write(line)

    output.close()


def initialize(data_path):
    global DRIVERS_TABLE, ROOT_DIR

    ROOT_DIR = data_path
    allfiles = listdir(data_path)
    DRIVERS_TABLE = {k: None for k in allfiles}
    
    random.seed(SEED)

    return allfiles


def do_adaboost(filename):
    df, Y = create_merged_dataset(filename)
    # Ideas:
    # Create a feature for accelerations e deacceleration.

    # Leave default base regressor for AdaBoost(decision tree). Extra trees were tried with catastrophic results.
    #ada = AdaBoostRegressor(n_estimators=350, learning_rate=0.05)
    ada = AdaBoostRegressor(n_estimators=500, learning_rate=1)
    
    #X = df.drop(['driver', 'trip', 'prob_points', 'prob_speed', 'prob_distance', 'prob_acceleration'], 1)
    X = df.drop(['driver', 'trip'], 1)
    ada.fit(X, Y)
    probs = ada.predict(X[:200])
    return pd.DataFrame({'driver': df['driver'][:200], 'trip': df['trip'][:200], 'probs': probs})


def do_logit(filename):
    df, Y = create_merged_dataset(filename)
    
    logit = LogisticRegression()
    X = df.drop(['driver', 'trip'], 1)
    logit.fit(X, Y)
    probs = [i[1] for i in logit.predict_proba(X[:200])]
    return pd.DataFrame({'driver': df['driver'][:200], 'trip': df['trip'][:200], 'probs': probs})


def do_etrees(filename):
    df, Y = create_merged_dataset(filename)
    etree = ExtraTreesRegressor(n_estimators=200, n_jobs=-1, min_samples_leaf=5, random_state=SEED)
    X = df.drop(['driver', 'trip'], 1)
    etree.fit(X, Y)
    probs = etree.predict(X[:200])
    return pd.DataFrame({'driver': df['driver'][:200], 'trip': df['trip'][:200], 'probs': probs})


def do_rf(filename):
    df, Y = create_merged_dataset(filename)
    rf = RandomForestRegressor(n_estimators=100)
    X = df.drop(['driver', 'trip'], 1)
    rf.fit(X, Y)
    probs = rf.predict(X[:200])
    return pd.DataFrame({'driver': df['driver'][:200], 'trip': df['trip'][:200], 'probs': probs})


def do_gbm(filename):
    df, Y = create_merged_dataset(filename)
    # Best result so far was achieved with this GBM parameters.
    gbm = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, min_samples_leaf=10, subsample=0.5)
    #X = df.drop(['driver', 'trip', 'prob_points', 'prob_speed', 'prob_distance', 'prob_acceleration'], 1)
    # With new set of features, just drop driver and trip.
    X = df.drop(['driver', 'trip'], 1)
    gbm.fit(X, Y)
    probs = gbm.predict(X[:200])
    return pd.DataFrame({'driver': df['driver'][:200], 'trip': df['trip'][:200], 'probs': probs})


def do_validation(data_path, steps=10):
    allfiles = initialize(data_path)
    gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, min_samples_leaf=5, subsample=0.5)
    ada = AdaBoostRegressor(n_estimators=200, learning_rate=1)
    etree = ExtraTreesRegressor(n_estimators=200, n_jobs=-1, min_samples_leaf=5)
    rf = RandomForestRegressor(n_estimators=200, max_features=4, min_samples_leaf=5)
    kn = KNeighborsRegressor(n_neighbors=25)
    logit = LogisticRegression(tol=0.05)
    enet = ElasticNetCV(l1_ratio=0.75, max_iter=1000, tol=0.05)
    svr = SVR(kernel="linear", probability=True)
    ridge = Ridge(alpha=18)
    bridge = BayesianRidge(n_iter=500)

    gbm_metrics = 0.0
    ada_metrics = 0.0
    etree_metrics = 0.0
    rf_metrics = 0.0
    kn_metrics = 0.0
    logit_metrics = 0.0
    svr_metrics = 0.0
    ridge_metrics = 0.0
    bridge_metrics = 0.0
    enet_metrics = 0.0
    nnet_metrics = 0.0

    logistic = LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose=True)
    classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

    for i in xrange(steps):
        driver = allfiles[i]
        df, Y = create_merged_dataset(driver)
        df['label'] = Y        
        # Shuffle DF.
        df = df.reindex(np.random.permutation(df.index))

        train = df[:100]
        label = train['label']
        del train['label']

        test = df[100:400]
        Y = test['label']
        del test['label']

        #to_drop = ['driver', 'trip', 'speed1', 'speed2', 'speed3', 'speed4', 'speed5', 'speed6', 'speed7', 'speed8', 'speed9', 
        #        'speed10', 'speed11', 'speed12', 'speed13', 'speed14', 'speed15', 'speed16', 'speed17', 'speed18', 'speed19', 
        #        'speed20', 'speed21', 'speed22', 'speed23', 'speed24', 'speed25', 'speed26', 'speed27', 'speed28', 'speed29', 
        #        'speed30', 'speed31', 'speed32', 'speed33', 'speed34', 'speed35', 'speed36', 'speed37', 'speed38', 'speed39', 
        #        'speed40', 'speed41', 'speed42', 'speed43', 'speed44', 'speed45', 'speed46', 'speed47', 'speed48', 'speed49', 
        #        'speed50', 'speed51', 'speed52', 'speed53', 'speed54', 'speed55', 'speed56', 'speed57', 'speed58', 'speed59', 
        #        'speed60', 'speed61', 'speed62', 'speed63', 'speed64', 'speed65', 'speed66', 'speed67', 'speed68', 'speed69', 
        #        'speed70', 'speed71', 'speed72', 'speed73', 'speed74', 'speed75', 'speed76', 'speed77', 'speed78', 'speed79', 'speed80']
        to_drop = ['driver', 'trip']

        X_train = train.drop(to_drop, 1)
        X_test = test.drop(to_drop, 1)
        
        gbm.fit(X_train, label)
        Y_hat = gbm.predict(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(Y, Y_hat)
        gbm_metrics += metrics.auc(fpr, tpr) 
        
        ada.fit(X_train, label)
        Y_hat = ada.predict(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(Y, Y_hat)
        ada_metrics += metrics.auc(fpr, tpr)
    
        etree.fit(X_train, label)
        Y_hat = etree.predict(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(Y, Y_hat)
        etree_metrics += metrics.auc(fpr, tpr)
        
        rf.fit(X_train, label)
        Y_hat = rf.predict(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(Y, Y_hat)
        rf_metrics += metrics.auc(fpr, tpr)
        
        kn.fit(X_train, label)
        Y_hat = kn.predict(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(Y, Y_hat)
        kn_metrics += metrics.auc(fpr, tpr)

        # Linear models.
        to_drop = ['driver', 'trip', 'distance', 'sd_acceleration', 'final_angle', 'mean_acceleration', 'mean_avg_speed', 'sd_inst_speed',
                'sd_avg_speed', 'mean_inst_speed', 'points']

        X_train = train.drop(to_drop, 1)
        X_test = test.drop(to_drop, 1)
        
        logit.fit(X_train, label)
        Y_hat = [i[1] for i in logit.predict_proba(X_test)]
        fpr, tpr, thresholds = metrics.roc_curve(Y, Y_hat)
        logit_metrics += metrics.auc(fpr, tpr)

        svr.fit(X_train, label)
        Y_hat = svr.predict(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(Y, Y_hat)
        svr_metrics += metrics.auc(fpr, tpr)
        
        ridge.fit(X_train, label)
        Y_hat = ridge.predict(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(Y, Y_hat)
        ridge_metrics += metrics.auc(fpr, tpr)

        bridge.fit(X_train, label)
        Y_hat = bridge.predict(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(Y, Y_hat)
        bridge_metrics += metrics.auc(fpr, tpr)

        enet.fit(X_train, label)
        Y_hat = enet.predict(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(Y, Y_hat)
        enet_metrics += metrics.auc(fpr, tpr)

        classifier.fit(X_train, label)
        Y_hat = classifier.predict(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(Y, Y_hat)
        nnet_metrics += metrics.auc(fpr, tpr)

    print ""
    print "GBM:", gbm_metrics/steps
    print "AdaBoost:", ada_metrics/steps
    print "Extra Trees:", etree_metrics/steps
    print "RF:", rf_metrics/steps
    print "KN:", kn_metrics/steps
    print ""
    print "Logit:", logit_metrics/steps
    print "SVR:", svr_metrics/steps
    print "Ridge:", ridge_metrics/steps
    print "BayesianRidge:", bridge_metrics/steps
    print "Elastic Net:", enet_metrics/steps
    print "Neural Networks:", nnet_metrics/steps
    print ""


def train_predict(data_path):
    print "Training and predicting..."

    allfiles = initialize(data_path)

    n = 0
    predictions = pd.DataFrame()

    # Read each driver trips
    for driver in allfiles:
        if n%250 == 0:
            print "n =", n

        probs = do_etrees(driver)
        
        predictions = predictions.append(probs, ignore_index=True)

        n += 1

    format_submission(predictions)


def do_param_search():
        allfiles = initialize(data_path)

        driver = allfiles[0]
        df, Y = create_merged_dataset(driver)
        df['label'] = Y        
        # Shuffle DF.
        df = df.reindex(np.random.permutation(df.index))

        train = df[:100]
        label = train['label']
        del train['label']

        test = df[100:400]
        Y = test['label']
        del test['label']

        to_drop = ['driver', 'trip']

        train = train.drop(to_drop, 1)
        test = test.drop(to_drop, 1)

        tuned_parameters = [{'n_estimators': [10, 20, 50, 100, 200, 300, 400], 'min_samples_split': [1, 2, 4, 6, 8, 10, 20],  
            'max_depth': [3, 6, 9, 12, None], 'min_samples_leaf': [1, 3, 6, 9, 12, 15, 20, 25], 'max_features': [1,2,3,4,5,6,7,8, None],
            'max_leaf_nodes': [2, 4, 6, 8, None], 'bootstrap': [True, False], 'oob_score': [True, False]}]
        gs = GridSearchCV(ExtraTreesRegressor(), tuned_parameters)
        gs.fit(train, label)

        print gs.best_estimator_


if __name__ == "__main__":
    data_path = "data/stats"
    seed = random.randrange(1, 2147483647)
    
    print "Seed:", seed
    SEED = seed

    if len(argv) > 1:
        if argv[1] == "-v":
            do_validation(data_path)
        if argv[1] == "-p":
            do_param_search()
    else:
        train_predict(data_path)

