# -*- coding: utf-8 -*-
#!/usr/bin/python
#
# Author    Yann Bayle
# E-mail    bayle.yann@live.fr
# License   MIT
# Created   03/10/2016
# Updated   08/11/2016
# Version   1.0.0
#

"""
Description of utils.py
======================

..todo:: 
pylint
"""

import os
import sys
import json
import random
import shutil
from datetime import datetime
from statistics import mean, stdev
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

VERBOSE = True

COLOR = {
    "HOUR" : "\033[96m",
    "HEADER" : "\033[95m",
    "OKBLUE" : "\033[94m",
    "OKGREEN" : "\033[92m",
    "WARNING" : "\033[93m",
    "ERROR" : "\033[91m",
    "FILE" : "\033[37m",
    "ENDC" : "\033[0m",
    "BOLD" : "\033[1m",
    "UNDERLINE" : "\033[4m"
}

def curTime():
    return COLOR["HOUR"] + datetime.now().time().strftime("%Hh%Mm%Ss") + " " + COLOR["ENDC"]

def print_error(msg):
    """Description of print_error

    Print error message and exit program
    """
    disp_msg = "\n" + curTime() + COLOR["BOLD"] + COLOR["ERROR"] + "ERROR:\n" 
    disp_msg = disp_msg + str(msg) + "\nProgram stopped" + COLOR["ENDC"]
    print(disp_msg)
    sys.exit()

def print_info(msg):
    """Description of print_info

    Print info message
    """
    if VERBOSE:
        print(curTime() + COLOR["OKBLUE"] + str(msg) + COLOR["ENDC"])

def print_warning(msg):
    """Description of print_warning

    Print warning message
    """
    if VERBOSE:
        print(curTime() + COLOR["WARNING"] + str(msg) + COLOR["ENDC"])

def print_success(msg):
    """Description of print_success

    Print success message
    """
    if VERBOSE:
        print(curTime() + COLOR["BOLD"] + COLOR["OKGREEN"] + msg + COLOR["ENDC"])

def print_msg(msg):
    """Description of print_msg

    Print default message
    """
    if VERBOSE:
        print(COLOR["HEADER"] + msg + COLOR["ENDC"])

def print_progress_start(msg):
    """Description of print_progress_start

    do not forget to call print_progress_end at the end 
    """
    if VERBOSE:
        print(curTime() + msg + COLOR["ENDC"])
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")

def print_progress_end():
    """Description of print_progress_end

    Print 
    """
    if VERBOSE:
        sys.stdout.write("\033[K")

def print_file(fileName):
    if os.path.isfile(fileName):
        printInfo(fileName + ":")
        print(COLOR["FILE"])
        with open(fileName, 'r') as fn:
            for line in fn:
                print(line[:-1])
        print(COLOR["ENDC"])
    else:
        printWarning("File not found: " + fileName)

def abs_path_dir(dir_name):
    """Description of abs_path_dir

    Check validity of directory and return absolute path
    Otherwise raise error and end programm
    """
    if not os.path.isfile(dir_name) and os.path.exists(dir_name):
        dir_name = os.path.abspath(dir_name)
        if dir_name[-1] == "/" or dir_name[-1] == "\\":
            return dir_name
        else:
            return dir_name + os.sep
    else:
        print_error("Invalid directory name: " + dir_name)

def abs_path_file(filename):
    """Description of abs_path_file

    Check validity of file name and return absolute path of it
    Otherwise raise error and end programm
    """
    if os.path.isfile(filename) and os.path.exists(filename):
        return os.path.abspath(filename)
    else:
        print_error("Invalid file name: " + filename)

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    dir_name = abs_path_dir(dir_name)
    if dir_name[-1] != "/" and dir_name[-1] != "\\":
        dir_name += os.sep
    return dir_name

def save_results(algo_name, predictions):
    res_dir = "res/"
    create_dir(res_dir)
    with open(res_dir + algo_name, "w") as filep:
        for pred in predictions:
            pass

def score_evolution_testset_size():
    pass

def scores(algo_name, predictions, groundtruths):
    """
    precision, recall, thresholds = precision_recall_curve(test_groundtruths, prediction)
    
    """    

    # # Init
    # data_file = "results.json"
    # if os.path.exists(data_file):
    #     with open(data_file, "r") as filep:    
    #         data = json.load(filep)
    # else:
    #     data = {}
    # data[algo_name] = {}
    # data[algo_name]["uneven"] = {}
    # data[algo_name]["balanced"] = {}
    # for distribution in data[algo_name]:
    #     data[algo_name][distribution]["precision"] = {}
    #     data[algo_name][distribution]["recall"] = {}
    #     data[algo_name][distribution]["f1"] = {}
    #     for tmp in data[algo_name][distribution]:
    #         data[algo_name][distribution][tmp]["instru"] = []
    #         data[algo_name][distribution][tmp]["song"] = []

    # # Unbalanced
    # # Song
    # data[algo_name]["uneven"]["precision"]["song"] = precision_score(groundtruths, predictions, average=None)[1]
    # data[algo_name]["uneven"]["recall"]["song"] = recall_score(groundtruths, predictions, average=None)[1]
    # data[algo_name]["uneven"]["f1"]["song"] = f1_score(groundtruths, predictions, average=None)[1]
    # # Instru
    # data[algo_name]["uneven"]["precision"]["instru"] = precision_score(groundtruths, predictions, average=None)[0]
    # data[algo_name]["uneven"]["recall"]["instru"] = recall_score(groundtruths, predictions, average=None)[0]
    # data[algo_name]["uneven"]["f1"]["instru"] = f1_score(groundtruths, predictions, average=None)[0]
    

    
    # Balanced
    instru_gts = []
    instru_pred = []
    song_gts = []
    song_pred = []
    song_tmp_gts = []
    song_tmp_pred = []
    cpt = 0
    nb_instru = groundtruths.count("i")
    for index, tag in enumerate(groundtruths):
        if "i" in groundtruths[index]:
            instru_gts.append("i")
            instru_pred.append(predictions[index])
        else:
            if cpt == nb_instru:
                song_gts.append(song_tmp_gts)
                song_pred.append(song_tmp_pred)
                song_tmp_gts = []
                song_tmp_pred = []
                cpt = 0
            else:
                song_tmp_gts.append("s")
                song_tmp_pred.append(predictions[index])
                cpt += 1

    acc = []
    f1 = []
    for index, row in enumerate(song_gts):
        groundtruths = instru_gts + song_gts[index]
        predictions = instru_pred + song_pred[index]
        # Song
        # data[algo_name]["balanced"]["precision"]["song"].append(precision_score(groundtruths, predictions, average=None)[1])
        # data[algo_name]["balanced"]["recall"]["song"].append(recall_score(groundtruths, predictions, average=None)[1])
        # data[algo_name]["balanced"]["f1"]["song"].append(f1_score(groundtruths, predictions, average=None)[1])
        # # Instru
        # data[algo_name]["balanced"]["precision"]["instru"].append(precision_score(groundtruths, predictions, average=None)[0])
        # data[algo_name]["balanced"]["recall"]["instru"].append(recall_score(groundtruths, predictions, average=None)[0])
        # data[algo_name]["balanced"]["f1"]["instru"].append(f1_score(groundtruths, predictions, average=None)[0])
        
        # new 14 december 2016 for new article
        acc.append(accuracy_score(groundtruths, predictions))
        f1.append(f1_score(groundtruths, predictions, average='weighted'))
        # f1.append(f1_score(groundtruths, predictions, average=None))
        # print(f1)

    # new 14 december 2016 for new article
    # Print average ± standard deviation
    print(algo_name)
    print("Accuracy " + str(sum(acc)/float(len(acc))) + " ± " + str(stdev(acc)))
    # print("F-Measure " + str(sum(f1)/float(len(f1))) + " ± " + str(stdev(f1)))

    # if ".csv" in algo_name:
    #     algo_name = algo_name[:-4]
    # if "'" in algo_name:
    #     algo_name = algo_name[:-1]

    # with open("../stats/table2_accuracy_R.csv", "a") as filep:
    #     for val in acc:
    #         filep.write(algo_name + "," + str(val) + "\n")
    # with open("../stats/table2_f1_R.csv", "a") as filep:
    #     for val in f1:
    #         filep.write(algo_name + "," + str(val) + "\n")
    # with open("../stats/table2_accuracy.csv", "a") as filep:
    #     filep.write(algo_name)
    #     for val in acc:
    #         filep.write("," + str(val))
    #     filep.write("\n")
    # with open("../stats/table2_f1.csv", "a") as filep:
    #     filep.write(algo_name)
    #     for val in f1:
    #         filep.write("," + str(val))
    #     filep.write("\n")

    # # Write 
    # with open(data_file, "w") as outfile:
    #     json.dump(data, outfile, indent=2)

def rand_color(nb=1):
    """Description of rand_color
    Generate an array of nb HTML color
    """
    if nb < 1:
        print_error("utils.rand_color(nb) requires a positive integer as nb")
    colors = []
    for index in range(0, nb):
        colors.append("#%06X" % random.randint(0,256**3-1))
    return colors

def read_groundtruths(filename):
    filename = abs_path_file(filename)
    groundtruths = {}
    with open(filename, "r") as filep:
        for line in filep:
            row = line[:-1].split(",")
            groundtruths[row[0]] = row[1]
    return groundtruths

def get_test_gts():
    test_groundtruths = {}
    with open("groundtruths.csv", "r") as filep:
        for row in filep:
            line = row.split(",")
            test_groundtruths[line[0]] = line[1][:-1]
    return test_groundtruths

def mv_files(fnames):
    tmp_dir = "/tmp/"
    with open(fnames, "r") as filep:
        for line in filep:
            fname = line.split("\n")[0]
            new_dir = fname.split(os.sep)
            new_dir = os.sep.join(new_dir[:-1]) + tmp_dir
            create_dir(new_dir)
            shutil.move(fname, new_dir + fname.split("/")[-1])
