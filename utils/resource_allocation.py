# ==================================================
# Copyright (C) 2017-2018
# author: yilin.shen
# email: yilin.shen@samsung.com
# Date: 2019-08-11
#
# This file is part of MRI project.
# 
# This can not be copied and/or distributed 
# without the express permission of yilin.shen
# ==================================================

# import pynvml in anaconda
import sys
import os
from collections import defaultdict

import importlib.machinery

anaconda3_path = os.path.abspath(sys.executable + "/../../")
pynvml_path = anaconda3_path + '/lib/python3.7/site-packages/'
sys.path.append(pynvml_path)

loader = importlib.machinery.SourceFileLoader('my_pynvml', pynvml_path + 'pynvml.py')
my_pynvml = loader.load_module()

import my_pynvml


def get_default_gpus(parallels):
    my_pynvml.nvmlInit()
    deviceCount = my_pynvml.nvmlDeviceGetCount()

    gpu_usage = defaultdict(int)
    for i in range(deviceCount):
        handle = my_pynvml.nvmlDeviceGetHandleByIndex(i)
        info = my_pynvml.nvmlDeviceGetMemoryInfo(handle)

        gpu_usage[i] = info.used

    sorted_gpu_usage = sorted(gpu_usage, key=gpu_usage.get)

    return sorted_gpu_usage[:parallels]
