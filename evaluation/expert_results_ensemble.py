# ==================================================
# Copyright (C) 2017-2018
# author: Claire Tang
# email: Claire Tang@gmail.com
# Date: 2019-08-23
#
# This file is part of MRI project.
# 
# This can not be copied and/or distributed 
# without the express permission of Claire Tang
# ==================================================

import csv
import numpy as np

result_in = 'results/ensemble_crop.csv'
result_out = 'results/ensemble.csv'

head = 'image,AK,BCC,BKL,DF,MEL,NV,SCC,VASC,UNK'


def combine_csv_avg():
    """average all cropped data"""

    result_dict = {}
    with open(result_in, 'r') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')

        for row in readCSV:

            if row[0] not in result_dict:
                result_dict[row[0]] = list()

            result_dict[row[0]].append([float(x) for x in row[1:]])

    with open(result_out, 'w') as f_out:
        f_out.write(head + '\n')

        for image, prob_list in result_dict.items():
            avg_prob = [str(sum(x) / 10) for x in zip(*prob_list)]
            row_output = [image] + avg_prob

            f_out.write(','.join(row_output) + '\n')


def combine_csv_voting():
    """majority voting of all cropped data"""

    result_dict = {}
    with open(result_in, 'r') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')

        for row in readCSV:

            if row[0] not in result_dict:
                result_dict[row[0]] = list()

            result_dict[row[0]].append([float(x) for x in row[1:]])

    with open(result_out, 'w') as f_out:
        f_out.write(head + '\n')

        for image, prob_list in result_dict.items():
            prob_list = np.array(prob_list)

            max_array = (prob_list == prob_list.max(axis=1)[:, None]).astype(float)

            sum_max_array = np.array([sum(x) for x in zip(*list(max_array))])

            voting_array = [0.0] * len(sum_max_array)
            if np.max(sum_max_array) <= 3.0:
                voting_array[8] = 1.0
            else:
                # voting_array[np.argmax(sum_max_array)] = 1.0
                voting_array = (sum_max_array == np.max(sum_max_array))
                voting_array = 1.0 * voting_array

            row_output = [image] + [str(float(x)) for x in list(voting_array)]

            f_out.write(','.join(row_output) + '\n')


combine_csv_voting()
