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
import classify
import argparse

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

def main(args):
    """
    @brief      Main entry point
    """
    train_fn = "train_fn.csv"
    merge_gt_feat(gt_filen=args.gtfn, feat_filen=args.featfn, train_filen=train_fn)
    classify.cross_validation(train_filename=train_fn, n_folds=2, outfilename="toto.csv")

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
