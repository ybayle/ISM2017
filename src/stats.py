# -*- coding: utf-8 -*-
#
# Author    Yann Bayle
# E-mail    bayle.yann@live.fr
# License   MIT
# Created   06/01/2017
# Updated   06/01/2017
# Version   1.0.0
#
"""
Description of stats.py
======================

1 Shapiro (test normality)
2 Bartlett (test homostedasticity)
3 if normal and same variance then ANOVA and Tukey post hoc test
  else kruskall wallis

TABLE 1
/media/sf_github/array1/repro.py : ghosal()
/media/sf_github/kea/kea.py figure2()
/media/sf_github/vqmm2/vqmm2.py figure1()

"""

from os import listdir
import sys
import utils
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statistics import stdev

def assert_normality(data):
    """Description of assert_normality
    The Shapiro-Wilk test tests the null hypothesis that the data was drawn 
    from a normal distribution
    if test_stat and p_value close to 1 then data is normal
    """
    print("Values " + str(data))
    statistic, pvalue = stats.shapiro(data)
    print("Shapiro Statistic " + str(statistic) + " and p-value " + str(pvalue))
    if pvalue > 0.05:
        print("Normal")
        return True
    else:
        print("Not normal")
        return False

def bartlett(data):
    """Description of bartlett
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bartlett.html
    """
    if len(data) == 2:
        statistic, pvalue = stats.bartlett(data[0], data[1])
    elif len(data) == 3:
        statistic, pvalue = stats.bartlett(data[0], data[1], data[2])
    elif len(data) == 4:
        statistic, pvalue = stats.bartlett(data[0], data[1], data[2], data[3])
    else:
        utils.print_error("TODO barlett manage more values")
    print("Bartlett Statistic " + str(statistic) + " and p-value " + str(pvalue))
    if pvalue > 0.05:
        return True
    else:
        return False

def assert_homoscedasticity(data):
    """Description of assert_homoscedasticity
    Bartlett’s test tests the null hypothesis that all input samples are from
    populations with equal variances. For samples from significantly non-normal
    populations, Levene’s test levene is used.
    """
    normality = True
    for item in data:
        normality = assert_normality(item)
        if not normality:
            break
    if normality:
        return bartlett(data)
    else:
        print("TODO levene")

def anova(data):
    """
    return True is at least one mean is different from the other
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html
    """
    if len(data) == 2:
        statistic, pvalue = stats.f_oneway(data[0], data[1])
    elif len(data) == 3:
        statistic, pvalue = stats.f_oneway(data[0], data[1], data[2])
    elif len(data) == 4:
        statistic, pvalue = stats.f_oneway(data[0], data[1], data[2], data[3])
    else:
        utils.print_error("TODO ANOVA manage more values")
    print("ANOVA Statistic " + str(statistic) + " and p-value " + str(pvalue))
    if pvalue < 0.05:
        return True
    else:
        return False

def tukey(data, names):
    names = np.array(names)
    tmp = []
    for item in data:
        for val in item:
            tmp.append(val)
    data = np.array(tmp)
    mc = MultiComparison(data, names)
    result = mc.tukeyhsd()
    print(result)
    print(mc.groupsunique)

def kruskal(data):
    """
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html
    """
    if len(data) == 2:
        statistic, pvalue = stats.kruskal(data[0], data[1])
    elif len(data) == 3:
        statistic, pvalue = stats.kruskal(data[0], data[1], data[2])
    elif len(data) == 4:
        statistic, pvalue = stats.kruskal(data[0], data[1], data[2], data[3])
    else:
        utils.print_error("TODO kruskal manage more values")
    print("Kruskal Statistic " + str(statistic) + " and p-value " + str(pvalue))
    if pvalue > 0.05:
        # same
        return False
    else:
        # different
        return True

def read_data_1(stats_dir, filen):
    stats_dir = utils.abs_path_dir(stats_dir)
    filen = utils.abs_path_file(filen)
    data = []
    names = []
    with open(stats_dir + filen, "r") as filep:
        for line in filep:
            # Read file with lines like this:
            # GA,0.578947368421,0.631578947368,0.710526315789,0.722222222222
            # SVMBFF,0.631578947368,0.684210526316,0.815789473684,0.66666666
            # VQMM,0.736842105263,0.842105263158,0.842105263158,0.75,0.61111
            row = line[:-1].split(",")
            tmp = []
            for index in range(1, len(row)):
                names.append(row[0])
                tmp.append(float(row[index]))
            data.append(tmp)
            print(filen.split(".")[0].split("_")[1].title() + " for " + row[0] + " \t= " + str("{0:.3f}".format(sum(tmp)/len(tmp))) + " ± " + str("{0:.3f}".format(stdev(tmp))))

def read_data_2(stats_dir, filen):
    stats_dir = utils.abs_path_dir(stats_dir)
    filen = utils.abs_path_file(stats_dir + filen)
    data = []
    names = []
    tmp = []
    name = ""
    with open(filen, "r") as filep:
        next(filep)
        for line in filep:
            row = line[:-1].split(";")
            if row[0] in names:
                names.append(row[0])
                tmp.append(float(row[1]))
            else:
                if len(tmp) > 0:
                    data.append(tmp)
                    tmp = []
                names.append(row[0])
                tmp.append(float(row[1]))
    data.append(tmp)
    return data, names

def main(stats_dir):
    utils.print_success("Statistical analysis")
    stats_file = listdir(utils.abs_path_dir(stats_dir))
    for filen in stats_file:
        utils.print_info(filen)
        data, names = read_data_2(stats_dir, filen)
        if assert_homoscedasticity(data):
            if anova(data):
                tukey(data, names)
            else:
                print("All means are the same")
        else:
            if kruskal(data):
                print("cf R")
                # Dunn
                # Conover-Iman
                # Dwass-Steel-Citchlow-Fligner
            else:
                print("All means are the same")

if __name__ == "__main__":
    main("../results/gender_old/")
