# -*- coding: utf-8 -*-
#!/usr/bin/python
#
# Author    Yann Bayle
# E-mail    bayle.yann@live.fr
# License   MIT
# Created   30/03/2017
# Updated   30/03/2017
# Version   1.0.0
#

"""
Description of recisio.py
======================


:Example:

source activate py27
python recisio.py

"""

import os
import re
import sys
import json
import utils
import classify

def add_feat_yaafe(dir_feat, data, ids=None):
    """
    @brief      Reads features files.
    
    @param      dir_feat  The folder containing the songs' features
    
    @return     dict (key=ids and values=features)
    """
    dir_feat = utils.abs_path_dir(dir_feat + "yaafe/")
    for filen in os.listdir(dir_feat):
        new_id = re.search(r"\d{3,9}", filen).group()
        if ids is None or new_id in ids:
            MFCCs = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
            with open(dir_feat + filen, "r") as filep:
                for _ in range(0, 5):
                    next(filep)
                for line_num, line in enumerate(filep):
                    row = line.split(",")
                    for index, val in enumerate(row):
                        MFCCs[index] += float(val)
            for index, mfcc in enumerate(MFCCs):
                MFCCs[index] = mfcc / (line_num + 1)
            if new_id in data:
                data[new_id].append(MFCCs)
            else:
                data[new_id] = MFCCs
    return data

def write2file(data, filen):
    """
    @brief      Writes train and test data in filen.
    
    @param      data   The data
    @param      filen  The filen
    
    @return     No return value
    """
    with open(filen, "a") as filep:
        for key in data:
            for mfcc in data[key]:
                filep.write(str(mfcc) + ",")
            filep.write(key + "\n")

def add_feat_essentia(dir_feat, data, ids=None):
    """
    @brief      Reads essentia features files.
    
    @param      dir_feat  The folder containing the songs' features
    
    @return     dict (key=ids and values=features)
    """
    dir_feat = utils.abs_path_dir(dir_feat + "essentia/")
    features = []
    for filen in os.listdir(dir_feat):
        new_id = re.search(r"\d{3,9}", filen).group()
        if ids is None or new_id in ids:
            with open(dir_feat + filen) as filep:
                essentia_feat = json.load(filep)

            features.append(essentia_feat["tonal"]["chords_changes_rate"])
            features.append(essentia_feat["tonal"]["chords_number_rate"])
            features.append(essentia_feat["tonal"]["tuning_frequency"])
            # features.append(essentia_feat["tonal"]["chords_key"])
            # features.append(essentia_feat["tonal"]["chords_scale"])
            # features.append(essentia_feat["tonal"]["key_key"])
            # features.append(essentia_feat["tonal"]["key_scale"])

            features.append(essentia_feat["rhythm"]["beats_count"])
            features.append(essentia_feat["rhythm"]["bpm"])
            features.append(essentia_feat["rhythm"]["danceability"])
            features.append(essentia_feat["rhythm"]["onset_rate"])
            
            # utils.print_error(features)
            if new_id in data:
                data[new_id].append(features)
            else:
                data[new_id] = features
    return data

def cover_song_identification():
    """
    @brief      Experiment which tries to identify the cover from an origin song
    
    @return     No return value
    """

    utils.print_success("Reading train and test files")

    # Load train files i.e. origin songs aka origins
    dir_feat_origins = "D:/_Doctorat/ISMIR2017/features/origins/"
    train = {}
    # train = add_feat_yaafe(dir_feat_origins, train)
    train = add_feat_essentia(dir_feat_origins, train)

    # Load test files i.e. cover song(s) aka covers
    dir_feat_covers = "E:/_These/DataSets/Recisio/features/"
    test = {}
    # test = add_feat_yaafe(dir_feat_covers, test, train)
    test = add_feat_essentia(dir_feat_origins, test, train)

    # filen = "test.csv"
    # write2file(train, filen)
    # write2file(test, filen)

    # Train / test and display results
    utils.print_success("Train / test and display results")
    classify.train_test(train, test)
    # classify.train_test(train, test={"12205":[1.5627507629177773,-0.02869418829824216,0.3018536332432878,0.07362493101783524,0.13370604355970842,0.055065859545780936,-0.002080898073517681,0.05894344517632047,0.019244777853631646,-0.02625581479272883,0.03434491230835797,0.04911969509928087,0.04703704763572111]})


def main():
    """
    @brief      Main entry point
    """
    cover_song_identification()

if __name__ == "__main__":
    main()
