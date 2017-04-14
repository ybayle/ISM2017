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
# import classify
import argparse
import soundfile as sf

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
    cpt()

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
