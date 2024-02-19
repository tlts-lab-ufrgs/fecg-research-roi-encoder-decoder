#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Haar generalized

Created on Wed Jul 19 20:55:39 2023

@author: julia
"""

#%% bib

import math
import numpy as np

#%% Haar Transform

class HaarTransform:
    
    def __init__(self, input_vec, decomp_level, norm_component):
        
        self.input_vec = input_vec
        
        self.decomposition_level = decomp_level
        
        self.norm_factor = norm_component
        
        self.inputs_parameters()
        
        self.haar_matrix, self.inv_haar_matrix = self.build_haar_matrix()
        
    def inputs_parameters(self):
        """
        Function to get input vectors 

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.input_vec_shape = np.shape(self.input_vec)
        
        # checks if is np array or, at least, a list
       
        if np.isscalar(self.input_vec) or type(self.input_vec) == 'string':
            raise Exception('Input vector must be an numpy array or a list, type given: %s' % (type(self.input_vec)))
        
        self.is_2D = True if len(self.input_vec_shape) > 1 else False
        
        if self.is_2D and (self.input_vec_shape[0] != self.input_vec_shape[1]):
            
            raise Exception('The input vector should have N x N dimensions if 2D!')
            
        # if 2D have to have N, N
            
        self.N = self.input_vec_shape[0]
        
        return 
    
    def build_haar_matrix(self, N = 0):
        """
        Build matrix T with detail and means 

        Returns
        -------
        None.

        """
        
        # If N is not given by user, use self.N
        N = self.N if N == 0 else N
        
        N = int(N)
        
        # Start matrix to be completed with values
        haar_matrix = np.zeros(shape = (N, N))
        
        j = 0
               
        for i in range(0, N):
            
            if i < N / 2 :
                
                haar_matrix[i, 2 * j] = 1
                haar_matrix[i, 2 * j + 1] = 1
                
            if i >= N / 2:
                
                if i == N/2:
                    
                    j = 0
                    
                haar_matrix[i, 2 * j] = 1
                haar_matrix[i, 2 * j + 1] = -1
                
            j += 1
            
        # Set the inverse
        
        matrix_with_norm = self.norm_factor * haar_matrix
        
        inv_matrix = np.linalg.inv(matrix_with_norm)
        
        return matrix_with_norm, inv_matrix
    
      
    def run_foward_transform(self):
        
        output_vector = np.zeros(shape=self.input_vec_shape)
        
        if self.is_2D:
            
            output_vector = self.haar_matrix @ self.input_vec @ self.inv_haar_matrix
            
        else:
            
            output_vector = self.haar_matrix @ self.input_vec 
        
        return output_vector 
    
    def run_inverse_transform(self):
        
        output_vector = np.zeros(shape=self.input_vec_shape)
        
        if self.is_2D:
            
            output_vector = self.inv_haar_matrix @ self.input_vec @ self.haar_matrix
            
        else:
            
            output_vector = self.inv_haar_matrix @ self.input_vec 
        
        return output_vector
    
    def build_multi_resolution_matrix(self, k):
        
        """
        Build H_{N, 2^k} matrix 
        -----------------------
        
        Parameters:
            k (int): iteration of producer, number of non null values
        
        
        """
     
        # Initialize H_{N, 2k} 
        partial_hn = np.zeros(shape = (self.N, self.N))
        
        k_squared = int(np.power(2, k))
               
        # Get T matrix
        haar_matrix, inv = self.build_haar_matrix(k_squared)
        
        
        # Complete upper right corner 
        partial_hn[
            0:k_squared, 0:k_squared
        ] = haar_matrix.copy()
        
        # Complete lower left corner
        
       
        # if int(self.N - np.power(2, k)) != 0: 
       
        partial_hn[
            k_squared:,  
            k_squared:
        ] = np.identity((self.N - k_squared))
            
        # print(partial_hn)
                       
        return partial_hn
        
    def run_cascade_multiresolution_transform(self):

        # Loop to build producer H_N'
               
        # Define start and stop of loop
        
        
        # loop from always will be bigger than loop_to
        
        loop_from = np.log2(self.N) + 1 - self.decomposition_level
        loop_to = np.log2(self.N) - 1
        
        # print(loop_from, loop_to)
        if loop_to < loop_from:
            
            producer_Hn = np.identity(self.N)
        
        else:
            i = loop_from
            
            while i <= loop_to:
                
                if i == loop_from:
                    producer_Hn = self.build_multi_resolution_matrix(i)
                 
                else:
                    partial_hn = self.build_multi_resolution_matrix(i)
                    
                    producer_Hn = np.matmul(producer_Hn, partial_hn)
                
                i += 1
        
        # Run decomposition for this level

        T_matrix, T_inv_matrix = self.build_haar_matrix(self.N)
        
    
        final_multi_matrix = np.matmul(producer_Hn, T_matrix)
        
        # print(final_multi_matrix)

        this_level_result = np.matmul(
            final_multi_matrix, 
            self.input_vec
            )
        
            
        return this_level_result
    
    def run_cascade_multiresolution_inv_transform(self):
        
    
        # loop from always will be bigger than loop_to
        
        loop_from = np.log2(self.N) + 1 - self.decomposition_level
        loop_to = np.log2(self.N) - 1
        
        # print(loop_from, loop_to)
        if loop_to < loop_from:
            
            producer_Hn = np.identity(self.N)
        
        else:
            i = loop_from
            
            while i <= loop_to:
                
                if i == loop_from:
                    producer_Hn = self.build_multi_resolution_matrix(i)
                 
                else:
                    partial_hn = self.build_multi_resolution_matrix(i)
                    
                    producer_Hn = np.matmul(producer_Hn, partial_hn)
                
                i += 1
        
        # Run decomposition for this level

        T_matrix, T_inv_matrix = self.build_haar_matrix(self.N)
        
    
        final_multi_matrix = np.linalg.inv(np.dot(producer_Hn, T_matrix))
        
        # print(final_multi_matrix)

        this_level_result = np.matmul(
            final_multi_matrix, 
            self.input_vec
            )
            
        return this_level_result
        
    

    def build_packet_multi_resolution_matrix(self, k):
        
        packet_multi_matrix = np.zeros(shape = (self.N, self.N))
        
        two_in_k = np.power(2, k)
        
        # Get T matrix
        haar_matrix = self.build_haar_matrix(np.power(2, k))
        
        i = 0
        
        while i < (self.N - two_in_k):
            
            packet_multi_matrix[
                    i : i + two_in_k, 
                    i : i + two_in_k
                ] = haar_matrix.copy()
            
            i += two_in_k
        
        
        return 
    
    @staticmethod
    def run_non_decimated_tranform(self):
        
        return 
    

# #%% parameters / inputs from user

# f = 1 / 2

# # f = 1 / np.sqrt(2)

# N = 4

# v = np.array([5.25, 0.75, 2, 0.5]) 


# levels = 2

# haar = HaarTransform(v, levels, f)

# #%% run transform

# # inverse_transform_array = haar.run_inverse_transform()


# foward_transform_array = haar.run_foward_transform()


# multi_resolution = haar.run_cascade_multiresolution_transform()

# #%%

# inv_multi = haar.run_cascade_multiresolution_inv_transform()


# #%%

# matrix, inv = haar.build_haar_matrix(N)

# # #%%

# # transform = np.dot(matrix, v)


# #%%

# x1 = np.arange(0, N, step = 1)
# x2 = np.arange(0, N, step=2)


# #%%

# fig, ax = plt.subplots(

# ax.plot(foward_transform_array)

# # #%%
# # ax.plot(x1, v, label='original signal')
# # ax.plot(
# #         x2,
# #         foward_transform_array[:int(N/2)], 
# #         label='transformed signal', 
# #         marker='o'
# #     )

# # ax.legend()