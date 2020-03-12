# ==================================================
# Copyright (C) 2017-2018
# author: Claire Tang
# email: Claire Tang@gmail.com
# Date: 2019-08-20
#
# This file is part of MRI project.
# 
# This can not be copied and/or distributed 
# without the express permission of Claire Tang
# ==================================================


# # reorder columns
# with open('results/%s' % result_csv, 'r') as infile, open('results/reordered_%s' % result_csv, 'a') as outfile:
#     # output dict needs a list for new column ordering
#     fieldnames = ['image', 'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
#     writer = csv.DictWriter(outfile, fieldnames=fieldnames)
#     # reorder the header first
#     writer.writeheader()
#     for row in csv.DictReader(infile):
#         # writes the reordered rows to the new file
#         writer.writerow(row)
