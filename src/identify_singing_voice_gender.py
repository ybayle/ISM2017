# -*- coding: utf-8 -*-
#!/usr/bin/python
#
# Author    Yann Bayle
# E-mail    bayle.yann@live.fr
# License   MIT
# Created   05/04/2017
# Updated   05/04/2017
# Version   1.0.0
#

"""
Description of identify_singing_voice_gender.py
======================

:Example:

python identify_singing_voice_gender.py -g groundtruths.csv -f features.csv

"""

import os
import re
import sys
import utils
import numpy as np
from yaafelib import *
import classify
import argparse
import soundfile as sf

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

def read_gts(gt_filen):
    """
    @brief      Parse ground truths file in a Python dictionary
    
    @param      gt_filen  The ground truths filename
    
    @return     Return a dictionary with key corresponding to songs' id and
                value corresponding to male or female tag
    """
    gt_filen = utils.abs_path_file(gt_filen)
    data = {}
    with open(gt_filen, "r") as filep:
        for line in filep:
            if "male," in line: # this works for ,male, or ,female,
                row = line.split(",")
                data[row[0]] = row[3]
    return data

def merge_gt_feat(gt_filen, feat_filen, train_filen):
    """
    @brief      Read the files containing ground truths and features and merge
                them to be used for classification
    
    @param      gt_filen    The ground truths filename
    @param      feat_filen  The features filename
    """
    utils.print_success("Adding groundtruth")
    feat_filen = utils.abs_path_file(feat_filen)
    gt_filen = utils.abs_path_file(gt_filen)
    gts = read_gts(gt_filen)
    output = open(train_filen, "w")
    with open(feat_filen, "r") as feat:
        cur_id = ""
        for line in feat:
            if "filename" in line:
                m = re.search(r"\d{2,10}", line)
                cur_id = m.group()
            elif len(cur_id) > 1 and "srate" not in line and cur_id in gts:
                output.write(str(cur_id) + "," + line[:-4] + gts[cur_id] + "\n")
    output.close()

def test_marsyas(args):
    """
    @brief      { function_description }
    
    @param      args  The arguments
    
    @return     { description_of_the_return_value }
    """
    train_fn = "train_fn.csv"
    merge_gt_feat(gt_filen=args.gtfn, feat_filen=args.featfn, train_filen=train_fn)
    # classify.cross_validation(train_filename=train_fn, n_folds=2, outfilename="toto.csv")

def error(error_name, file_with_error):
    with open("errors.txt", "a") as filep:
        filep.write(error_name + "\t" + file_with_error + "\n")

def generate_singing_voice_track(paths):
    """
    @brief      { function_description }
    
    @param      dir_audio  The dir audio
    
    @return     { description_of_the_return_value }
    """
    utils.print_success("Generating singing voice tracks")
    for index, folder in enumerate(paths):
        utils.print_progress_start(str(index) + "/" + str(len(paths)) + " " + folder)
        if os.path.isdir(folder) and os.path.exists(folder):
            filelist = os.listdir(folder)
            for filen in filelist:
                nb_error = 0
                if "-bv." in filen:
                    utils.print_error("Backing vocals file found in " + filen)
                if filen.endswith('-ld.wav'):
                    song = filen
                    instru = filen.replace("-ld", "")
                    try:
                        song_samples, song_fs = sf.read(folder + "/" + song)
                    except RuntimeError as run_err:
                        error("RuntimeError", str(run_err))
                        nb_error += 1
                    try:
                        instru_samples, instru_fs = sf.read(folder + "/" + instru)
                    except RuntimeError as run_err:
                        error("RuntimeError", str(run_err))
                        nb_error += 1
                    if nb_error == 0:
                        if song_fs != instru_fs:
                            error("SamplingFreq", filen)
                        elif len(instru_samples) != len(song_samples):
                            error("SampleSize", filen)
                        else:
                            voice_samples = song_samples - instru_samples
                            # print(instru)
                            # print(song)
                            # utils.print_error(len(song_samples))
                            # print(song_samples.shape)
                            # print(len(instru_samples))
                            # print(instru_samples.shape)
                            # print(len(voice_samples))
                            # print(voice_samples.shape)
                            sf.write(folder + "/" + filen.replace("nbv-ld", "sv"), voice_samples, song_fs)
                            with open("available.txt", "a") as filep:
                                filep.write(folder + "/" + filen + "\n")
    utils.print_progress_end()
    return "available.txt"

def create_filelist(kara1k, dir_audio):
    """
    @brief      Creates a filelist.
    
    @param      kara1k  The kara 1 k
    
    @return     { description_of_the_return_value }
    """
    utils.print_success("Creating file list to be analyzed")
    kara1k = utils.abs_path_file(kara1k)
    dir_audio = utils.abs_path_dir(dir_audio)
    filelist = []
    with open(kara1k, "r") as filep:
        next(filep)
        for line in filep:
            row = line.split(",")
            # "male" in line avoir females and male and get female and male tracks
            if "1" in row[6] and "1" in row[7] and "1" in row[8] and "male" in line:
                filelist.append(dir_audio + row[1] + "_" + row[2] + "_" + row[0])
    return filelist

def extract_features(files="available.txt"):
    utils.print_success("Extracting features")
    files = utils.abs_path_file(files)
    dir_feat = "/media/sf_DATA/ISMIR2017/features/gender/"
    with open(files, "r") as filep:
        for line in filep:
            line = line[:-1]
            utils.yaafe(line, dir_feat + "song/", verbose=True)
            utils.yaafe(line.replace("nbv-ld", "sv"), dir_feat + "sv/", verbose=True)

def cpt():
    ids= []
    female = 0
    male = 0
    with open("gender.txt", "r") as filep:
        for line in filep:
            ids.append(re.search(r"\d{3,9}", line).group())
    with open("../data/filelist.csv", "r") as filep:
        next(filep)
        for line in filep:
            cur_id = re.search(r"\d{3,9}", line).group()
            if cur_id in ids:
                if ",female," in line:
                    female += 1
                elif ",male," in line:
                    male += 1
    print("female " + str(female))
    print("male " + str(male))

def float2str(numpy_float_arr):
    return ["%.8f" % number for number in numpy_float_arr]

def save_feat(data, out_filen, verbose=False):
    features = []
    fp = FeaturePlan(sample_rate=22050, normalize=True, resample=True)
    fp.addFeature('mfcc: MFCC blockSize=2048 stepSize=1024')
    fp.addFeature('mfcc_d1: MFCC blockSize=2048 stepSize=1024 > Derivate DOrder=1')
    fp.addFeature('mfcc_d2: MFCC blockSize=2048 stepSize=1024 > Derivate DOrder=2')
    df = fp.getDataFlow()
    if verbose:
        df.display()
    engine = Engine()
    engine.load(fp.getDataFlow())
    engine.reset() # first reset the engine

    engine.writeInput('audio', data) # write audio array on 'audio' input
    engine.process() # process available data
    feats = engine.readAllOutputs() # read available feature data
    features.append(feats)
    engine.flush() # do not forget to flush
    with open(out_filen, "w") as filep:
        for mfcc, mfcc_d1, mfcc_d2 in zip(features[0]["mfcc"], features[0]["mfcc_d1"], features[0]["mfcc_d2"]):
            filep.write(",".join(float2str(mfcc)) + "," + ",".join(float2str(mfcc_d1)) + "," + ",".join(float2str(mfcc_d2)) + "\n")

def remove_silence(filen, verbose=False):
    utils.print_success("Removing silence")
    filen = utils.abs_path_file(filen)
    with open(filen, "r") as filep:
        for line in filep:
            print(line)
            # Step 1 Gather samples
            song_fn = line[:-1]
            voice_fn = song_fn.replace("nbv-ld", "sv")
            try:
                voice_samples, voice_fs = sf.read(voice_fn)
            except:
                utils.print_error("ERROR in identify_singing_voice_gender line 207 in sf.read(voice_fn)")
            try:
                song_samples, song_fs = sf.read(song_fn)
            except:
                utils.print_error("ERROR in identify_singing_voice_gender line 207 in sf.read(song_fn)")
            idxs = np.any(voice_samples != 0., axis=1) # index of rows with at least one non zero value
            voice_samples_non_zero = voice_samples[idxs, 0] # selection of the wanted rows
            voice_samples_non_zero = voice_samples_non_zero.reshape(1, len(voice_samples_non_zero))
            song_samples_non_zero = song_samples[idxs, 0] # selection of the wanted rows
            song_samples_non_zero = song_samples_non_zero.reshape(1, len(song_samples_non_zero))
            
            # Step 2 Extract features
            dir_feat = "/media/sf_DATA/ISMIR2017/features/gender/"
            save_feat(voice_samples_non_zero, dir_feat + "sv_nonzero/" + voice_fn.split(os.sep)[-1] + ".mfcc")
            save_feat(song_samples_non_zero, dir_feat + "song_nonzero/" + song_fn.split(os.sep)[-1] + ".mfcc")

def get_gts():
    gt_filen = "../data/filelist.csv"
    utils.abs_path_file(gt_filen)
    gts = {}
    with open(gt_filen, "r") as filep:
        next(filep)
        for line in filep:
            if ",male," in line or ",female," in line:
                row = line.split(",")
                gts[row[0]] = row[3][0]
    return gts

def add_gts2files(gts, folder):
    filelist = os.listdir(folder)
    for filen in filelist:
        cur_gt = gts[re.search(r"\d{3,9}", filen).group()]
        print(filen + " " + cur_gt)
        data = ""
        with open(folder + filen, "r") as filep:
            for line in filep:
                data += line.replace("\n", "," + cur_gt + "\n")
        with open(folder + filen, "w") as filep:
            filep.write(data)

def add_groundtruths(folder):
    utils.print_success("Adding groundtruths")
    gts = get_gts()
    add_gts2files(gts, utils.abs_path_dir(folder + "song_nonzero/"))
    add_gts2files(gts, utils.abs_path_dir(folder + "sv_nonzero/"))

def merge_files(folder, name):
    utils.print_success("Merging files")
    subfolder = utils.abs_path_dir(folder + name)
    data = ""
    for filen in os.listdir(subfolder):
        with open(subfolder + filen, "r") as filep:
            for line in filep:
                data += line
    with open(folder + name + ".csv", "w") as filep:
        filep.write(data)

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
        "RandomForest": RandomForestClassifier(n_jobs=-1)
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
    return predictions

def cross_validation(train_filename, n_folds, outfilename):    
    utils.print_success("Cross validation")
    filename = utils.abs_path_file(train_filename)

    condition = train_filename.split(".")[0].split(os.sep)[-1]

    features = []
    groundtruths = []
    with open(filename, "r") as filep:
        for line in filep:
            line = line[:-1].split(",")
            features.append([float(x) for x in line[0:-1]])
            groundtruths.append(line[-1])
    features = np.array(features)
    groundtruths = np.array(groundtruths)

    skf = StratifiedKFold(n_splits=n_folds)
    # for i in range(0, 10):
    i = 0
    cur_fold = 0

    with open("../results/gender/precision.txt", "a") as filep:
        filep.write(condition + ";" + str(precision_score(dataset["test_groundtruths"], predictions, average='weighted')) + "\n")
    with open("../results/gender/recall.txt", "a") as filep:
        filep.write(condition + ";" + str(recall_score(dataset["test_groundtruths"], predictions, average='weighted')) + "\n")
    with open("../results/gender/f1.txt", "a") as filep:
        filep.write(condition + ";" + str(f1_score(dataset["test_groundtruths"], predictions, average='weighted')) + "\n")
    with open("../results/gender/accuracy.txt", "a") as filep:
        filep.write(condition + ";" + str(accuracy_score(dataset["test_groundtruths"], predictions)) + "\n")
    for train, test in skf.split(features, groundtruths):
        cur_fold += 1
        utils.print_success("Iteration " + str(i) + "\tFold " + str(cur_fold))
        dataset = {}
        dataset["train_features"] = features[train]
        dataset["train_groundtruths"] = groundtruths[train]
        dataset["test_features"] = features[test]
        dataset["test_groundtruths"] = groundtruths[test]
        predictions = classify(data=dataset)

        print("\tPrecision weighted\t" + str(precision_score(dataset["test_groundtruths"], predictions, average='weighted')))
        print("\tRecall weighted\t" + str(recall_score(dataset["test_groundtruths"], predictions, average='weighted')))
        print("\tF1 weighted\t" + str(f1_score(dataset["test_groundtruths"], predictions, average='weighted')))
        print("\tAccuracy\t" + str(accuracy_score(dataset["test_groundtruths"], predictions)))
        with open("../results/gender/precision.txt", "a") as filep:
            filep.write(str(precision_score(dataset["test_groundtruths"], predictions, average='weighted')) + "\n")
        with open("../results/gender/recall.txt", "a") as filep:
            filep.write(str(recall_score(dataset["test_groundtruths"], predictions, average='weighted')) + "\n")
        with open("../results/gender/f1.txt", "a") as filep:
            filep.write(str(f1_score(dataset["test_groundtruths"], predictions, average='weighted')) + "\n")
        with open("../results/gender/accuracy.txt", "a") as filep:
            filep.write(str(accuracy_score(dataset["test_groundtruths"], predictions)) + "\n")

def main(args):
    """
    @brief      Main entry point
    Singer Gender Identification
        Extract features (yaafe)
            voice
            song
        Experiment 1 : 5-CV train voice Test voice
        Experiment 2 : 5-CV train song Test song
        Experiment 3 : train song + voice Test song
        Experiment 4 : train song + voice Test Simbals
        IEEE : train on good frames for instru male female
    """
    utils.print_success("Starting SGI")
    # dir_audio = "E:/_These/DataSets/Recisio/audio/"
    # dir_audio = "/media/sf_SharedFolder/DataSets/Recisio/audio/"
    # kara1k = "../data/filelist.csv"
    # paths = create_filelist(kara1k, dir_audio)
    # available_files = generate_singing_voice_track(paths)
    # extract_features(available_files)
    # extract_features()
    # cpt()
    # remove_silence("gender.txt")
    folder = "/media/sf_DATA/ISMIR2017/features/gender/"
    # add_groundtruths(folder)
    # merge_files(folder, "song_nonzero")
    # merge_files(folder, "sv_nonzero")
    n_folds = 10
    outfilename = folder + "sv_nonzero_results.txt"
    cross_validation(folder + "sv_nonzero.csv", n_folds, outfilename)
    outfilename = folder + "song_nonzero_results.txt"
    cross_validation(folder + "song_nonzero.csv", n_folds, outfilename)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Classify the singing voice's gender")
    PARSER.add_argument(
        "-g",
        "--gtfn",
        help="The ground truths filename",
        type=str,
        default="D:/_Doctorat/ISMIR2017/data/filelist.csv",
        metavar="gtfn")
    PARSER.add_argument(
        "-f",
        "--featfn",
        help="The features filename",
        type=str,
        default="D:/_Doctorat/ISMIR2017/features/marsyas.arff",
        metavar="featfn")
    main(PARSER.parse_args())
