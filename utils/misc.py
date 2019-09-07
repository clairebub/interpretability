# https://github.com/yenchanghsu/out-of-distribution-detection/blob/master/utils/misc.py

import torch
import torch.nn as nn


def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.
    From https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


# def print_dict_table(tbl):
#     metric_names = tbl['Gaussian'].keys()  # Our implementation always has Gaussian in the ood benchmark
#     print('| ' + 'ood dataset'.ljust(20), end='')
#     for n in metric_names:
#         print('| ' + n.ljust(10), end='')
#     print('')
#     for d, perf in tbl.items():
#         print('| ' + d.ljust(20), end='')
#         for m in metric_names:
#             if isinstance(perf[m], str):
#                 print('| ' + perf[m].ljust(10), end='')
#             else:
#                 print('| ' + str(perf[m])[:6].ljust(10), end='')
#         print('')
