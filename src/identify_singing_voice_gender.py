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
import sys
import utils
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

def merge_gt_feat(gt_filen, feat_filen):
    """
    @brief      Read the files containing ground truths and features and merge
                them to be used for classification
    
    @param      gt_filen    The ground truths filename
    @param      feat_filen  The features filename
    """
    utils.print_success("Adding groundtruth")
    feat_filen = utils.abs_path_file(feat_filen)
    gt_filen = utils.abs_path_file(gt_filen)
    groundtruths = read_gts(gt_filen)
    output_fn = "gender_feat_gt.csv"
    output = open(output_fn, "w")

    # switch if test set preprocessing
    # separator = "_"
    separator = "."

    with open(feat_filen, "r") as feat:
        line_num = 0
        tmp_line = ""
        for line in feat:
            line_num += 1
            if line_num > 74:
                if line[0] != "%":
                    # Alter feature line with correct tag
                    cur_line = line.split(",")
                    old_tag = cur_line[-1].split(separator)[0]
                    if old_tag in groundtruths:
                        new_tag = groundtruths[old_tag]
                        output.write(tmp_line + ",".join(cur_line[:-1]) + "," + new_tag +"\n")
                        tmp_line = ""
                        tags.append(new_tag)
                    else:
                        # File not in groundtruth
                        tmp_line = ""
                        # utils.print_warning("Error with " + old_tag)
                else:
                    tmp_line += line
            elif line_num == 2:
                output.write("@relation train_test.arff\n")
                # output.write("@relation MARSYAS_KEA\n")
            elif line_num == 71:
                # Alter line 71 containing all tag gathered along the way
                output.write("@attribute output {i,s}\n")
            else:
                # Write header
                output.write(line)
    output.close()

def main(args):
    """
    @brief      Main entry point
    """
    merge_gt_feat(gt_filen=args.gtfn, feat_filen=args.featfn)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Classify the singing voice's gender")
    PARSER.add_argument(
        "-g",
        "--gtfn",
        help="The ground truths filename",
        type=str,
        default="D:/_Doctorat/ISMIR2017/data/recisio.csv",
        metavar="gtfn")
    PARSER.add_argument(
        "-f",
        "--featfn",
        help="The features filename",
        type=str,
        default="D:/_Doctorat/ISMIR2017/features/marsyas.arff",
        metavar="featfn")
    main(PARSER.parse_args())
