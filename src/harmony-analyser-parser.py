# -*- coding: utf-8 -*-
#!/usr/bin/python
#
# Author    Yann Bayle
# E-mail    bayle.yann@live.fr
# License   MIT
# Created   12/04/2017
# Updated   12/04/2017
# Version   1.0.0
#

"""
Description of harmony-analyser-parser.py
======================

:Example:

python harmony-analyser-parser.py

"""

import os
import re
import sys
import utils
import shutil
import argparse

def extract_features(in_dir, out_dir, path, verbose=False):
    in_dir = utils.abs_path_dir(in_dir)
    path = utils.abs_path_dir(path)
    cur_dir = os.getcwd()
    os.chdir(in_dir)
    script = "harmony-analyser-script-jar-with-dependencies.jar"
    src = path + script
    dst = in_dir + script
    shutil.copy(src, dst)
    options = [
        "nnls-chroma:nnls-chroma",
        "nnls-chroma:chordino-tones",
        "nnls-chroma:chordino-labels",
        "qm-vamp-plugins:qm-keydetector",
        "chord_analyser:tps_distance"
        # "chord_analyser:chord_complexity_distance",
        # "chroma_analyser:complexity_difference",
        # "chord_analyser:average_chord_complexity_distance"
    ]
    for opt in options:
        cmd = "java -jar " + script + " -a " + opt + " -s .wav -t 0.07"
        utils.run_cmd(cmd, verbose)
    os.remove(dst)
    cp_cmd = "cp *.txt " + out_dir
    utils.run_cmd(cp_cmd)
    # utils.run_cmd("rm *.txt")
    os.chdir(cur_dir)

def main(args):
    """
    @brief      Main entry point
    """
    path = utils.abs_path_dir(args.path)
    in_dir = utils.abs_path_dir(args.in_dir)
    out_dir = utils.abs_path_dir(args.out_dir)
    id_songs_feat_done = []
    for filen in os.listdir(out_dir):
        if os.path.isfile(out_dir + filen):
            m = re.search(r"\d{3,9}", filen)
            id_songs_feat_done.append(m.group())
    id_songs_feat_done = list(set(id_songs_feat_done))
    index = 0
    with open("../data/filelist.csv", "r") as filep:
        for line in filep:
            row = line[:-1].split(",")
            # Check if features have been extracted by YAAFE, Marsyas & Essentia
            if "1" in row[6] and "1" in row[7] and "1" in row[8]:
                if not row[0] in id_songs_feat_done:
                    folder = in_dir + row[1] + "_" + row[2] + "_" + row[0]
                    index += 1
                    print(str(index) + " " + folder)
                    extract_features(folder, out_dir, path)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Parser for harmony-analyser")
    PARSER.add_argument(
        "-i",
        "--in_dir",
        help="The intput directory containing wave files",
        type=str,
        default="/media/sf_SharedFolder/DataSets/Recisio/audio/",
        metavar="in_dir")
    PARSER.add_argument(
        "-o",
        "--out_dir",
        help="The output directory for storing features",
        type=str,
        default="/media/sf_DATA/ISMIR2017/features/origins/chromas/",
        metavar="out_dir")
    PARSER.add_argument(
        "-p",
        "--path",
        help="The path for the harmony-analyser script",
        type=str,
        default="/home/yann/Documents/harmony-analyser/harmony-analyser/target/",
        metavar="path")
    main(PARSER.parse_args())
