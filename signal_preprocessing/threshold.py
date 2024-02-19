#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 10:10:17 2023

@author: julia
"""

import numpy as np



def filtering_coef(N, details, level, type_threshold, type_thresholding):
    
    if type_threshold == 'minimax':
        threshold = minimax_threshold(N, details)
    elif type_threshold == 'universal':
        threshold = universal_threshold(N, details)
    elif type_threshold == 'han':
        threshold = han_etal_threshold(N, details, level)
    elif type_threshold == 'alice':
        threshold = spc_shrink(details)
        

    if type_thresholding == 'hard':
        return_value = details
        
    elif type_thresholding == 'soft':
        return_value = (details / np.abs(details)) * (
            np.abs(details) - threshold
        )
        
    details = np.where(
        np.abs(details) >= threshold, 
        return_value, 
        0
    )
    
    return details


def minimax_threshold(N, details):
    
    threshold = sigma(details) * (0.3936 + 0.1829 * np.log2(N))
    
    return threshold

def universal_threshold(N, details):
    
    threshold = sigma(details) * np.sqrt(2 * np.log(N))
    
    return threshold

def han_etal_threshold(N, details, level, L=10):
    
    if level == 1:
        threshold = sigma(details) * np.sqrt(2 * np.log(N))
    elif level > 1 and level < L:
        threshold = sigma(details) * np.sqrt(2 * np.log(N)) / np.log(level + 1)
    elif level == L:
        threshold = sigma(details) * np.sqrt(2 * np.log(N)) / np.sqrt(level)
    
    return threshold

def spc_shrink(details, coef_d = 2.6):  #2.432
    
    # padrao alpha = 1%
    
    # alice treshholding
    
    data = np.copy(details)
    
    keep_calculating = True
    
    N = len(data)
    mean_wavelet = np.mean(data)

    while keep_calculating:
        
        deviation_s = np.sqrt(
                (1 / (N - 1)) * np.sum(np.square(data - mean_wavelet))
            )
        
        LCL = - coef_d * deviation_s
        UCL = coef_d * deviation_s
    
        data = data[(data >= LCL) & (data <= UCL)]
    
        if len(data) == N:
            keep_calculating = False
        else:
            N = len(data)
    
    return coef_d * deviation_s

def sigma(details):
    
    mean_detail = np.median(details)
    
    details -= mean_detail
    
    sigma = 1.4826 * np.median(np.abs(details))
    
    return sigma