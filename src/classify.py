# -*- coding: utf-8 -*-
#!/usr/bin/python
#
# Author    Yann Bayle
# E-mail    bayle.yann@live.fr
# License   MIT
# Created   03/11/2016
# Updated   11/12/2016
# Version   1.0.0
#

"""
Description of classify.py
======================

save train & test in files
read train & test
for list of classifier
    train/test
    gather results prec/rec/f
print best clf and results

:Example:

source activate py27
ipython
run classify.py

Only for 100 percent precision
run classify.py --train /media/sf_DATA/Datasets/Simbals/yann/train.csv --test /media/sf_DATA/Datasets/Simbals/yann/test.csv

notes
RandomForest complexity
https://www.quora.com/What-is-in-general-time-complexity-of-random-forest-What-are-the-important-parameters-that-affect-this-complexity
n instances and m attributes
computational cost of building a tree is O(mn log n).
RandomForest done in 135939ms (3mn) for 13 attributes and 192 instances
    mn log n = 13*192*math.log(192) = 13122 ( 135939ms)
    mn log n = 39*186*math.log(186) = 37907 (~ms)

To know the element available
print((clf.get_params().keys())

..todo::
Add
AdaBoostClassifier
BaggingClassifier
BernoulliNB
CalibratedClassifierCV
DPGMM
    http://scikit-learn.org/stable/modules/generated/sklearn.mixture.DPGMM.html
    Deprecated since version 0.18: This class will be removed in 0.20.
    Use sklearn.mixture.BayesianGaussianMixture with parameter
    weight_concentration_prior_type='dirichlet_process' instead.

DecisionTreeClassifier
ExtraTreeClassifier
ExtraTreesClassifier
GMM
GaussianNB
GradientBoostingClassifier
KNeighborsClassifier
LDA
LabelPropagation
LabelSpreading
LinearDiscriminantAnalysis
LogisticRegression
LogisticRegressionCV
MultinomialNB
NuSVC
QDA
QuadraticDiscriminantAnalysis
RandomForestClassifier
SGDClassifier
SVC
VBGMM
_ConstantPredictor

"""

import os
import sys
import time
import json
import utils
import joblib
import argparse
import webbrowser
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from statistics import stdev
from functools import partial
from sklearn import metrics
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, classification_report, f1_score, accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn import linear_model
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score

def list_clf():
    """
    ..todo::
    Do the same for:
        class_weight
        predict
        predict_log_proba
    """
    estimators = all_estimators()

    for name, class_ in estimators:
        if hasattr(class_, 'predict_proba'):
            print(name)

def plot_clf(indir="res/"):
    indir = utils.abs_path_dir(indir) + "/"
    algos = []
    measure = []
    with open(indir + "global.csv", "r") as filep:
        for line in filep:
            line = line.split(",")
            algos.append(line[0])
            measure.append(tuple(map(float, line[1:4])))

    n_groups = 3
    fig, ax = plt.subplots(figsize=(10, 6))

    index = np.arange(n_groups)
    bar_width = 0.2

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    color = utils.rand_color(len(algos))
    rects = {}
    offset = 0.15
    for ind, algo in enumerate(algos):
        print(ind)
        print(tuple(measure[ind]))
        rects[ind] = plt.bar(index + bar_width*ind + offset, tuple(measure[ind]), bar_width,
                     alpha=opacity,
                     color=color[ind],
                     label=algo)

    plt.ylabel('Scores (in %)')
    plt.xticks(index + bar_width*ind + offset, ('Precision', 'Recall', 'F-Measure'))
    plt.legend()
    plt.ylim(0, 1)

    # spines & axis
    ax = plt.gca()
    ax.spines['right'].set_color('none') 
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    art = []
    lgd = ax.legend(loc=9, bbox_to_anchor=(1.1, 1.), frameon=False)
    # lgd = pylab.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
    art.append(lgd)
    # ax.legend()
    plt.tight_layout()
    img_name = "global.png"
    plt.savefig(img_name, dpi=200, additional_artists=art, bbox_inches="tight")
    # webbrowser.open(img_name)
    # plt.show()

def read_file(filename):
    """Description of read_file

    train/test example line:
    filename,feat1,feat2,...,featn,tag
    """
    filename = utils.abs_path_file(filename)
    groundtruths = []
    features = []
    with open(filename, "r") as filep:
        for row in filep:
            line = row.split(",")
            groundtruths.append(line[-1][:-1])
            features.append([float(i) for i in line[1:-1]])
    return features, groundtruths

def read_preds(filename):
    """Description of read_file

    ex file:
    ISRC,tag
    """
    filename = utils.abs_path_file(filename)
    isrcs = {}
    with open(filename, "r") as filep:
        for row in filep:
            line = row.split(",")
            # print(line)
            isrcs[line[0]] = float(line[1])
            # isrcs[line[0]] = 1.0-float(line[1])
    return isrcs

def read_item_tag(filename):
    """Description of read_file

    example line:
    filename,tag
    """

    filename = utils.abs_path_file(filename)
    groundtruths = {}
    with open(filename, "r") as filep:
        for row in filep:
            line = row.split(",")
            groundtruths[line[0]] = line[1][:-1]
    return groundtruths

def precision_100percent(train, test):
    """Description of precision_100percent

    ..todo::
        1 Find best clf with default param
        2 vary param of best clf and find best param
        3 use best param and best clf to find recall for 100 percent precision
    """
    utils.print_success("Find Recall for best Precision for each tag")
    train = utils.abs_path_file(train)
    test = utils.abs_path_file(test)
    train_features, train_groundtruths = read_file(train)
    test_features, test_groundtruths = read_file(test)
    classifiers = {
        # "RandomForest": RandomForestClassifier(),#n_estimators=5
        "DecisionTree":DecisionTreeClassifier()#,#max_depth=10
        # "SVM":SVC(kernel="linear", C=0.0205),
        # "ExtraTreesClassifier":ExtraTreesClassifier(n_estimators=5, criterion="entropy", max_features="log2", max_depth=9),
        # "LogisticRegression":LogisticRegression()
    }
    tags = list(set(test_groundtruths))
    nb_tag = len(tags)
    step = 0.01
    # for index, tag in enumerate(["i"]):
    for index, tag in enumerate(tags):
        utils.print_success("Tag " + tag)
        max_precision = 0
        max_recall = 0
        max_f_measure = 0
        max_clf = ""
        max_weight = 0
        for key in classifiers:
            clf = classifiers[key]
            # for weight in np.arange(0., 0.01, 0.000001):
            # for weight in np.arange(step, 1-step, step):
            for weight in np.arange(0.0, 1.0, step):
                print("Classifier " + key + " & Weight " + str(weight))
                sys.stdout.write("\033[F")
                sys.stdout.write("\033[K")
                clf.set_params(class_weight={"i":weight, "s":1-weight})
                clf.fit(train_features, train_groundtruths)
                predictions = clf.predict(test_features)
                precision = precision_score(test_groundtruths, predictions, average=None)[index]
                if precision >= max_precision:
                    recall = recall_score(test_groundtruths, predictions, average=None)[index]
                    # if recall > max_recall:
                    max_precision = precision
                    max_recall = recall
                    max_f_measure = f1_score(test_groundtruths, predictions, average=None)[index]
                    max_weight = weight
                    max_clf = key
        sys.stdout.write("\033[K")
        utils.print_info("\tClassifier " + str(max_clf))
        utils.print_info("\tPrecision  " + str(max_precision))
        utils.print_info("\tRecall     " + str(max_recall))
        utils.print_info("\tF-Measure  " + str(max_f_measure))
        utils.print_info("\tWeight     " + str(max_weight))

def classify(train=None, test=None, data=None, res_dir="res/", disp=True, outfilename=None):
    """Description of compare
    compare multiple classifier and display the best one
    """
    utils.print_success("Comparison of differents classifiers")
    if data is not None:
        train_features = data["train_features"]
        train_groundtruths = data["train_groundtruths"]
        test_features = data["test_features"]
        test_groundtruths = data["test_groundtruths"]
    else:
        train = utils.abs_path_file(train)
        test = utils.abs_path_file(test)
        train_features, train_groundtruths = read_file(train)
        test_features, test_groundtruths = read_file(test)
    if not utils.create_dir(res_dir):
        res_dir = utils.abs_path_dir(res_dir)
    classifiers = {
        "RandomForest": RandomForestClassifier()
        # "RandomForest": RandomForestClassifier(n_estimators=5),
        # "KNeighbors":KNeighborsClassifier(3),
        # "GaussianProcess":GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
        # "DecisionTree":DecisionTreeClassifier(max_depth=5),
        # "MLP":MLPClassifier(),
        # "AdaBoost":AdaBoostClassifier(),
        # "GaussianNB":GaussianNB(),
        # "QDA":QuadraticDiscriminantAnalysis(),
        # "SVM":SVC(kernel="linear", C=0.025),
        # "GradientBoosting":GradientBoostingClassifier(),
        # "ExtraTrees":ExtraTreesClassifier(),
        # "LogisticRegression":LogisticRegression(),
        # "LinearDiscriminantAnalysis":LinearDiscriminantAnalysis()
    }
    for key in classifiers:
        utils.print_success(key)
        clf = classifiers[key]
        utils.print_info("\tFit")
        clf.fit(train_features, train_groundtruths)
        utils.print_info("\tPredict")
        predictions = clf.predict(test_features)

        if outfilename is not None:
            with open(outfilename, "w") as filep:
                for gt, pred in zip(test_groundtruths, predictions):
                    filep.write(gt + "," + pred + "\n")

        # Global
        data = [key]
        data.append(str(precision_score(test_groundtruths, predictions, average='weighted')))
        data.append(str(recall_score(test_groundtruths, predictions, average='weighted')))
        data.append(str(f1_score(test_groundtruths, predictions, average='weighted')))
        data = ",".join(data)
        if disp:
            print(data)
        else:
            with open(res_dir + "global.csv", "a") as filep:
                filep.write(data + ",\n")
        # Local
        for index, tag in enumerate(list(set(train_groundtruths))):
            precision = precision_score(test_groundtruths, predictions, average=None)
            recall = recall_score(test_groundtruths, predictions, average=None)
            f1 = f1_score(test_groundtruths, predictions, average=None)
            line = key + "," + str(precision[index]) + "," + str(recall[index]) + "," + str(f1[index])
            if disp:
                print(line)
            else:
                with open(res_dir + "tag_" + tag + ".csv", "a") as filep:
                    filep.write(line + ",\n")
    return predictions

def read_train_files(indir, separator=" "):
    """Description of read_train_files

    Gather local features and GT from every individual train songs
    """
    utils.print_success("Reading multiple train files")
    indir = utils.abs_path_dir(indir) + "/"
    groundtruths = []
    features = []
    included_extenstions = ["csv"]
    filenames = [fn for fn in os.listdir(indir)
            if any(fn.endswith(ext) for ext in included_extenstions)]
    for index, filename in enumerate(filenames):
        print(str(index + 1) + "/" + str(len(filenames)) + " " + filename)
        sys.stdout.write("\033[F") # Cursor up one line       
        sys.stdout.write("\033[K") # Clear line
        with open(indir + filename, "r") as filep:
            for row in filep:
                line = row.split(separator)
                features.append([float(i) for i in line[:-1]])
                groundtruths.append(line[-1][:-1])
    sys.stdout.write("\033[K") # Clear line
    return features, groundtruths

def read_train_file(filename):
    """
    Read ONE train file
    """
    groundtruths = []
    features = []
    filename = utils.abs_path_file(filename)
    with open(filename, "r") as filep:
        for line in filep:
            line = line.split(",")
            groundtruths.append(line[-1][:-1])
            features.append(line[1:-1])
    return features, groundtruths

def create_model(clf_name, features, groundtruths, outdir, classifiers):
    begin = int(round(time.time() * 1000))
    utils.print_success("Starting " + clf_name)
    clf_dir = outdir + clf_name + "/"
    utils.create_dir(clf_dir)
    clf = classifiers[clf_name]
    clf.fit(features, groundtruths)
    joblib.dump(clf, clf_dir + clf_name + ".pkl")
    utils.print_info(clf_name + " done in " + str(int(round(time.time() * 1000)) - begin) + "ms")

def create_models(outdir, train_features=None, train_groundtruths=None, train_file=None, train_dir=None, separator=" ", classifiers=None):
    """Description of create_models

    Generate models for train data for different clf
    In order to test later

    ..notes::
    train_file must be formatted like:

    item_name_1,feat1,feat2,...,featN,tag_or_class
    item_name_2,feat1,feat2,...,featN,tag_or_class
    ...
    item_name_N,feat1,feat2,...,featN,tag_or_class

    ..todo::
        Manage when Provide train feat and gts or train_file
        Find why commented clf cannot be used

    pour train dir = /media/sf_github/yann/train/
        20h04m49s Creating models
        20h04m49s Reading multiple train files
        20h05m04s Starting SVM
        20h05m07s Starting RandomForest
        20h05m11s Starting GradientBoosting
        20h05m16s Starting DecisionTree
        20h05m22s Starting ExtraTrees
        20h05m27s Starting AdaBoost
        20h05m34s Starting KNeighbors
        20h05m50s KNeighbors done in 60836ms
        20h06m18s ExtraTrees done in 89147ms
        20h06m29s DecisionTree done in 100211ms
        20h07m05s RandomForest done in 135939ms
        20h08m56s AdaBoost done in 246550ms
        20h13m40s GradientBoosting done in 530909ms
        00h43m29s SVM done in 16719954ms
    """

    utils.print_success("Creating models")

    outdir = utils.abs_path_dir(outdir) + "/"

    if train_file is not None:
        features, groundtruths = read_train_file(train_file)
    elif train_dir is not None:
        features, groundtruths = read_train_files(train_dir, separator=separator)
    else:
        utils.print_warning("TODO Manage train feat and gts")

    if classifiers is None:
        classifiers = {
            "RandomForest": RandomForestClassifier(),
            "LogisticRegression":LogisticRegression(),
            "KNeighbors":KNeighborsClassifier(),
            "DecisionTree":DecisionTreeClassifier(),
            "AdaBoost":AdaBoostClassifier(),
            "GradientBoosting":GradientBoostingClassifier(),
            "ExtraTrees":ExtraTreesClassifier(),
            "SVM":SVC(kernel="linear", C=0.025, probability=True)

            # "GaussianProcess":GaussianProcessClassifier(),
            # "MLP":MLPClassifier(),
            # "GaussianNB":GaussianNB(),
            # "QDA":QuadraticDiscriminantAnalysis(),
            # "LinearDiscriminantAnalysis":LinearDiscriminantAnalysis()
        }
    else:
        if "RandomForest" in classifiers:
            clf_name = "RandomForest"
            begin = int(round(time.time() * 1000))
            utils.print_success("Starting " + clf_name)
            clf_dir = outdir + clf_name + "/"
            utils.create_dir(clf_dir)
            clf = RandomForestClassifier(n_jobs=-1)
            # clf = RandomForestClassifier(verbose=100)
            clf.fit(features, groundtruths)
            joblib.dump(clf, clf_dir + clf_name + ".pkl")
            utils.print_info(clf_name + " done in " + str(int(round(time.time() * 1000)) - begin) + "ms")

    # # Parallel computing
    # clf = []
    # for key in classifiers:
    #     clf.append(key)
    # partial_create_model = partial(create_model, features=features, groundtruths=groundtruths, outdir=outdir, classifiers=classifiers)
    # # pool = multiprocessing.Pool(4)
    # pool = multiprocessing.Pool(len(classifiers))
    # pool.map(partial_create_model, clf) #make our results with a map call
    # pool.close() #we are not adding any more processes
    # pool.join() #tell it to wait until all threads are done before going on

def read_test_file(filename):
    """
    Read ONE test file with content like:

        feat1 feat2 ... featN
        feat1 feat2 ... featN
        ...
        feat1 feat2 ... featN

    """
    features = []
    filename = utils.abs_path_file(filename)
    with open(filename, "r") as filep:
        for line in filep:
            line = line.split(" ")
            line[-1] = line[-1][:-1]
            feat = []
            for tmp_feat in line:
                feat.append(float(tmp_feat))
            features.append(feat)
    return features

def column(matrix, i):
    return [row[i] for row in matrix]

def test_models(models_dir, test_dir, out_dir):
    models_dir = utils.abs_path_dir(models_dir) + "/"
    test_dir = utils.abs_path_dir(test_dir) + "/"
    utils.create_dir(out_dir)
    test_files = os.listdir(test_dir)
    models = os.listdir(models_dir)
    for model in models:
        utils.print_success(model)
        pred_dir = out_dir + model + "/"
        utils.create_dir(pred_dir)
        clf = joblib.load(models_dir + model + "/" + model + ".pkl")
        for index, test_file in enumerate(test_files):
            print(str(index) + "\t" + test_file)
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
            test_features = read_test_file(test_dir + test_file)
            predictions = clf.predict_proba(test_features)
            with open(pred_dir + test_file, "w") as filep:
                for pred in predictions:
                    filep.write(str(pred[0]) + "\n")
        sys.stdout.write("\033[K")

def test_model(model, models_dir, test_dir, out_dir, test_files=None, test_file=None):
    """Description of test_model

    Use one model previously fitted in order to predict_proba() or predict()
    the tag for a bunch of test_files 

    ..todo::
    
    To enhance computation time: only compute file which are in groundtruths
    if file already computed, do not recompute

    """
    begin = int(round(time.time() * 1000))
    utils.print_success("Testing " + model)
    pred_dir = out_dir + model
    clf = joblib.load(models_dir + model + "/" + model + ".pkl")
    if test_files is not None:
        pred_dir = pred_dir + "/"
        utils.create_dir(pred_dir)
        for index, test_file in enumerate(test_files):
            # Check if isrc is in groundtruths to speed up computation time
            if test_file[:12] in groundtruths:
                test_features = read_test_file(test_dir + test_file)
                try:
                    predictions = clf.predict_proba(test_features)
                except AttributeError:
                    utils.print_warning("predict_proba does not exists for " + model + "\nRegular predict function is used.")
                    predictions = clf.predict(test_features)
                with open(pred_dir + test_file, "w") as filep:
                    for pred in predictions:
                        filep.write(str(pred[0]) + "\n")
    elif test_file is not None:
        pred_dir = pred_dir + "_"
        test_features = []
        filename = []
        with open(test_file, "r") as filep:
            for index, line in enumerate(filep):
                line = line.split(",")
                # print(str(index) + " " + line[0])
                test_features.append(line[1:-1])
                filename.append(line[0])
        try:
            predictions = clf.predict_proba(test_features)
            with open(pred_dir + "predict_proba.csv", "a") as filep2:
                for filen, pred in zip(filename, predictions):
                    filep2.write(filen + "," + str(pred[0]) + "\n")
        except:
            pass
        predictions = clf.predict(test_features)
        with open(pred_dir + "predict.csv", "a") as filep2:
            for filen, pred in zip(filename, predictions):
                filep2.write(filen + "," + str(pred[0]) + "\n")
    else:
        utils.print_error("Error in arg for test_model() function")
    utils.print_info(model + " done in " + str(int(round(time.time() * 1000)) - begin) + "ms")

def test_models_parallel(models_dir, out_dir, test_dir=None, test_file=None):
    """Description of test_models_parallel

    17h16m12s DecisionTree done in 16135373ms
    17h25m08s GradientBoosting done in 16671109ms
    18h59m05s RandomForest done in 22307811ms
    18h59m07s AdaBoost done in 22310633ms
    19h18m12s ExtraTrees done in 23455779ms

    """
    models_dir = utils.abs_path_dir(models_dir) + "/"
    models = os.listdir(models_dir)
    utils.create_dir(out_dir)

    if test_dir is not None:
        test_dir = utils.abs_path_dir(test_dir) + "/"
        test_files = os.listdir(test_dir)
        test_file = None
    elif test_file is not None:
        test_files = None
    else:
        utils.print_warning("TODO Error in arg for test_models_parallel() function")

    partial_test_model = partial(test_model, models_dir=models_dir, test_dir=test_dir, out_dir=out_dir, test_files=test_files, test_file=test_file)
    pool = multiprocessing.Pool(len(models))
    pool.map(partial_test_model, models) #make our results with a map call
    pool.close() #we are not adding any more processes
    pool.join() #tell it to wait until all threads are done before going on

def cross_validation(train_filename, n_folds, outfilename):    

    filename = utils.abs_path_file(train_filename)
    features = []
    groundtruths = []
    with open(filename, "r") as filep:
        for line in filep:
            line = line.split(",")
            features.append([float(x) for x in line[1:-1]])
            groundtruths.append(line[-1][:-1])
    features = np.array(features)
    groundtruths = np.array(groundtruths)

    # Init    
    # if os.path.exists(outfilename):
    try:
        with open(outfilename, "r") as filep:   
            data = json.load(filep)
    except:
        data = {}
    # else:
    #     data = {}
    algo_name = "Method 1"
    data[algo_name] = {}
    data[algo_name]["uneven"] = {}
    data[algo_name]["balanced"] = {}
    for distribution in data[algo_name]:
        data[algo_name][distribution]["precision"] = {}
        data[algo_name][distribution]["recall"] = {}
        data[algo_name][distribution]["f1"] = {}
        for tmp in data[algo_name][distribution]:
            data[algo_name][distribution][tmp]["instru"] = []
            data[algo_name][distribution][tmp]["song"] = []

    skf = StratifiedKFold(n_splits=n_folds)
    for i in range(0, 10):
        utils.print_warning("TODO for i in range")
        song_precis = []
        song_recall = []
        song_fmeasu = []
        inst_precis = []
        inst_recall = []
        inst_fmeasu = []
        cur_fold = 0
        for train, test in skf.split(features, groundtruths):
            cur_fold += 1
            utils.print_success("Iteration " + str(i) + "\tFold " + str(cur_fold))
            dataset = {}
            dataset["train_features"] = features[train]
            dataset["train_groundtruths"] = groundtruths[train]
            dataset["test_features"] = features[test]
            dataset["test_groundtruths"] = groundtruths[test]
            predictions = classify(data=dataset)

            song_precis.append(precision_score(dataset["test_groundtruths"], predictions, average=None)[1])
            song_recall.append(recall_score(dataset["test_groundtruths"], predictions, average=None)[1])
            song_fmeasu.append(f1_score(dataset["test_groundtruths"], predictions, average=None)[1])
            inst_precis.append(precision_score(dataset["test_groundtruths"], predictions, average=None)[0])
            inst_recall.append(recall_score(dataset["test_groundtruths"], predictions, average=None)[0])
            inst_fmeasu.append(f1_score(dataset["test_groundtruths"], predictions, average=None)[0])

        song_precis = sum(song_precis) / float(len(song_precis))
        song_recall = sum(song_recall) / float(len(song_recall))
        song_fmeasu = sum(song_fmeasu) / float(len(song_fmeasu))
        inst_precis = sum(inst_precis) / float(len(inst_precis))
        inst_recall = sum(inst_recall) / float(len(inst_recall))
        inst_fmeasu = sum(inst_fmeasu) / float(len(inst_fmeasu))

        # Song
        data[algo_name]["balanced"]["precision"]["song"].append(song_precis)
        data[algo_name]["balanced"]["recall"]["song"].append(song_recall)
        data[algo_name]["balanced"]["f1"]["song"].append(song_fmeasu)
        # Instru
        data[algo_name]["balanced"]["precision"]["instru"].append(inst_precis)
        data[algo_name]["balanced"]["recall"]["instru"].append(inst_recall)
        data[algo_name]["balanced"]["f1"]["instru"].append(inst_fmeasu)

    with open(outfilename, "w") as outfile:
        json.dump(data, outfile, indent=2)

def split(features, groundtruths, n_split):
    """Description of split

    1 tmp array containing all item for each tag
    2 random split of array for each tag

    ..todo::

    manage possible errors
    randomize split selection

    """

    if n_split == 1:
        return features, groundtruths

    tags = list(set(groundtruths))
    new_index = {}
    for tag in tags:
        new_index[tag] = []
    for index, gt in enumerate(groundtruths):
        new_index[gt].append(index)
    new_feats = []
    new_gts = []
    for i in range(0, n_split):
        indexes = []
        for tag in tags:
            ref = len(new_index[tag])/n_split
            indexes.append(new_index[tag][ref*i:ref*(i+1)])
        """
        ..todo:: manage multiple tags!
        """
        indexes = indexes[0] + indexes[1]
        # print(features[:5])
        # print(len(indexes))
        # print(len(indexes[0]))
        # print(len(indexes[1]))
        # sys.exit()
        indexes.sort()
        new_gts.append([groundtruths[j] for j in indexes])
        new_feats.append([features[j] for j in indexes])
    return new_feats, new_gts

def increasing_test(groundtruths_file, predictions_file, metric, tag):
    gts = read_item_tag(groundtruths_file)
    preds = read_item_tag(predictions_file)
    
    test_groundtruths = []
    predictions = []
    for isrc in preds:
        if isrc in gts:
            test_groundtruths.append(gts[isrc]) 
            predictions.append(preds[isrc]) 

    res = []
    if "accuracy" in metric:
        res.append(accuracy_score(test_groundtruths, predictions))
    elif "precision" in metric:
        res.append(precision_score(test_groundtruths, predictions, average=None)[tag])
    elif "recall" in metric:
        res.append(recall_score(test_groundtruths, predictions, average=None)[tag])
    elif "f1_score" in metric:
        res.append(f1_score(test_groundtruths, predictions, average=None)[tag])
    else:
        utils.print_error("classify.py line 735 metric argument error")
    # print("Accuracy : " + str(accuracy_score(test_groundtruths, predictions)))
    # print("Precision: " + str(precision_score(test_groundtruths, predictions, average=None)))
    # print("Recall   : " + str(recall_score(test_groundtruths, predictions, average=None)))
    # print("F-score  : " + str(f1_score(test_groundtruths, predictions, average=None)))

    n_splits = 10
    # for n_split in range(2, n_splits+1):
    for n_split in [2, 10, 100]:
        print("\t" + str(n_split))
        feats_array, gts_array = split(predictions, test_groundtruths, n_split)
        tmp_acc = []
        for feats, gts in zip(feats_array, gts_array):
            if "accuracy" in metric:
                cur_acc = accuracy_score(gts, feats)
            elif "precision" in metric:
                cur_acc = precision_score(gts, feats, average=None)[tag]
            elif "recall" in metric:
                cur_acc = recall_score(gts, feats, average=None)[tag]
            elif "f1_score" in metric:
                cur_acc = f1_score(gts, feats, average=None)[tag]
            tmp_acc.append(cur_acc)
        print("\t\t" + str(stdev(tmp_acc)))
        accuracy = sum(tmp_acc) / float(len(tmp_acc))
        res.append(accuracy)
    return res

def growing_testset(train_filename, test_filename, clf, clf_name=None):
    """Description of growing_testset

    1 Generate accuracy graph for global
    2 Create precision / recall / f-measure figures for each tag

    ..todo::
    
    intermediate file which stores predictions for each ISRC
    param for number of steps
    repet N times
    division problem ! it does N N/2 ... N/10 but we want :
    1*N/10 2*N/10 ... 10*N/10 

    """
    train_features, train_groundtruths = read_file(train_filename)
    test_features, test_groundtruths = read_file(test_filename)
    
    if clf_name is not None and "RANSAC" in clf_name:
        train_groundtruths = [True if i =="s" else False for i in train_groundtruths]
        test_groundtruths = [True if i =="s" else False for i in test_groundtruths]

    clf.fit(train_features, train_groundtruths)

    if clf_name is not None and "RANSAC" in clf_name:
        preds_float = clf.predict(test_features)
        predictions = [True if i > 0.5 else False for i in preds_float]
    else:
        predictions = clf.predict(test_features)
    
    test_acc = []
    # test_acc.append(accuracy_score(test_groundtruths, predictions))
    test_acc.append(precision_score(test_groundtruths, predictions, average=None)[0])
    print("Accuracy : " + str(test_acc))
    print("Precision: " + str(precision_score(test_groundtruths, predictions, average=None)))
    print("Recall   : " + str(recall_score(test_groundtruths, predictions, average=None)))
    print("F-score  : " + str(f1_score(test_groundtruths, predictions, average=None)))

    n_splits = 10
    for n_split in range(2, n_splits+1):
        print(n_split)
        feats_array, gts_array = split(test_features, test_groundtruths, n_split)
        tmp_acc = []
        for feats, gts in zip(feats_array, gts_array):
            if clf_name is not None and "RANSAC" in clf_name:
                preds_float = clf.predict(feats)
                predictions = [True if i > 0.5 else False for i in preds_float]
            else:
                predictions = clf.predict(feats)
            # cur_acc = accuracy_score(gts, predictions)
            cur_acc = precision_score(gts, predictions, average=None)[0]
            tmp_acc.append(cur_acc)
            print("\t" + str(cur_acc))
        accuracy = sum(tmp_acc) / float(len(tmp_acc))
        test_acc.append(accuracy)
    return test_acc

def plot_roc(indir, gts_file, outdir):
    groundtruths = read_item_tag(gts_file)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', label="Random (0.5)")
    
    indir = utils.abs_path_dir(indir)
    for item in os.listdir(indir):
        if ".csv" in item:
            isrcs = read_preds(indir + "/" + item)
            test_groundtruths = []
            predictions = []
            for isrc in isrcs:
                if isrc in groundtruths:
                    test_groundtruths.append(groundtruths[isrc])
                    predictions.append(isrcs[isrc])
            test_groundtruths = [tag=="s" for tag in test_groundtruths]
            fpr_rf, tpr_rf, _ = roc_curve(test_groundtruths, predictions)
            label = item[:-4] + " (" + str(round(roc_auc_score(test_groundtruths, predictions), 3)) + ")"
            color = ""
            if "VQMM" in item:
                color = "ro"
            elif "SVMBFF" in item:
                color = "g-"
            elif "GA" in item:
                color = "b:"
            
            plt.plot(fpr_rf, tpr_rf, color, label=label)

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    # plt.title('ROC curve for Algo (AUC)')
    plt.legend(loc='best')
    outdir = utils.abs_path_dir(outdir)
    roc_fn = outdir + "Figure_3_ROC.png"
    plt.savefig(roc_fn, dpi=200, bbox_inches="tight")
    plt.savefig(outdir + "Figure_3_ROC.eps")
    # plt.show()
    plt.close()
    utils.print_success("ROC curve successfully created in " + roc_fn)

def plot_precision_recall(indir, gts_file, outdir):
    groundtruths = read_item_tag(gts_file)
    plt.figure(1)
    
    indir = utils.abs_path_dir(indir)
    for item in os.listdir(indir):
        if ".csv" in item:
            isrcs = read_preds(indir + "/" + item)
            test_groundtruths = []
            predictions = []
            for isrc in isrcs:
                if isrc in groundtruths:
                    test_groundtruths.append(groundtruths[isrc])
                    predictions.append(isrcs[isrc])
            test_groundtruths = [tag=="s" for tag in test_groundtruths]
            precision, recall, _ = precision_recall_curve(test_groundtruths, predictions)
            plt.plot(recall, precision, label=item[:-4] + " (" + str(round(average_precision_score(test_groundtruths, predictions), 3)) + ")")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.title('Precision-Recall curve for Algo (AUC)')
    plt.legend(loc='best')
    plt.savefig(outdir + "precision_recall.png", dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()
    utils.print_success("Precision-Recall curve created in " + outdir)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Compare classifiers")
    PARSER.add_argument(
        "--train",
        help="path to train file",
        type=str,
        default="data/proba_hist_train.csv",
        metavar="train")
    PARSER.add_argument(
        "--test",
        help="path to test file",
        type=str,
        default="data/proba_hist_test.csv",
        metavar="test")
    PARSER.add_argument(
        "-o",
        "--outdir",
        help="path to output directory",
        type=str,
        default="res/",
        metavar="outdir")
    
    plot_roc("roc_curve/")
    # plot_precision_recall("/media/sf_github/classifiers/roc_curve/")

    # # models_dir = "models_paral/"
    # # utils.create_dir(models_dir)

    # # train_file_1 = "/media/sf_DATA/Datasets/Simbals/yann/train.csv"
    # # train_dir_1 = "/media/sf_github/yann/train/"
    # # create_models(train_file=train_file_1)
    # # create_models(outdir=models_dir, train_dir=train_dir_1)
    # # test_models_parallel(models_dir, "/media/sf_DATA/Datasets/Simbals/yaafe/results/processed/", "/media/sf_DATA/Datasets/Simbals/yaafe/proba_preds/")

    # # classify(PARSER.parse_args().train, PARSER.parse_args().test, PARSER.parse_args().outdir)
    # # precision_100percent(PARSER.parse_args().train, PARSER.parse_args().test)
    # #  plot_clf()

    # """
    # Samedi 26 Novembre 2016 test finaux pour mon algo
    # demandé par Pierre et Matthias
    # """
    # train_file = "/media/sf_github/yann/2_local_predictions/method_3_trainset_normalized.txt"
    # models_dir = "final_models/"
    # utils.create_dir(models_dir)
    # # create_models(outdir=models_dir, train_file=train_file)

    # out_dir = "/media/sf_DATA/Datasets/Simbals/yann/algo_final/"
    # utils.create_dir(out_dir)
    # test_file="/media/sf_github/yann/2_local_predictions/method_3_testset_normalized_with_tag.txt"
    # # test_models_parallel(
    # #     models_dir=models_dir,
    # #     test_file=test_file,
    # #     out_dir=out_dir)
    
    # test_features = []
    # isrc_order = []
    # utils.print_info("Loading clf")
    # clf = joblib.load("/media/sf_github/classifiers/final_modelsRandomForest/RandomForest.pkl")
    # with open(test_file, "r") as filep:
    #     for index, line in enumerate(filep):
    #         line = line.split(",")
    #         utils.print_info(str(index) + "\t" + line[0])
    #         test_features.append(line[1:-1])
    #         isrc_order.append(line[0])
    # utils.print_info("Predict_proba")
    # predictions = clf.predict(test_features)
    # # predictions = clf.predict_proba(test_features)
    # utils.print_info("Writing results")
    # with open("/media/sf_DATA/Datasets/Simbals/yann/algo_final/RF.txt" , "w") as filep2:
    #     for index, pred in enumerate(predictions):
    #         filep2.write(isrc_order[index] + "," + str(pred[0]) + "\n")
    # utils.print_info("Done")

    # test_groundtruths = {}
    # with open("/media/sf_github/repro/groundtruths.csv", "r") as filep:
    #     for row in filep:
    #         line = row.split(",")
    #         test_groundtruths[line[0]] = line[1][:-1]
    # for i in np.arange(0.1, 1.0, 0.1):
    #     outfile = open("results/Bayle2_"+str(i)+".csv", "w")
    #     utils.print_progress_start(str(i))
    #     with open("/media/sf_DATA/Datasets/Simbals/yann/algo_final/RFproba.txt", "r") as filep:
    #         for line in filep:
    #             line = line.split(",")
    #             if line[0] in test_groundtruths:
    #                 if float(line[-1][:-1]) > i:
    #                     prediction = "i"
    #                 else:
    #                     prediction = "s"
    #                 outfile.write(line[0] + "," + prediction + "\n")
    #     utils.print_progress_end()
    #     outfile.close()

    # # groundtruths = []
    # # predictions = []
    # outfile = open("results/Bayle.csv", "w")
    # with open("/media/sf_DATA/Datasets/Simbals/yann/algo_final/RF.txt", "r") as filep:
    #     for line in filep:
    #         line = line.split(",")
    #         if line[0] in test_groundtruths:
    #             outfile.write(line[0] + "," + line[-1][:-1] + "\n")
    #             # groundtruths.append(test_groundtruths[line[0]])
    #             # predictions.append(line[-1][:-1])
    # outfile.close()
    # # utils.scores("bayle", predictions, groundtruths)
