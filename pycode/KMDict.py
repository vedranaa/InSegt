#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:53:41 2021

@author: abda
"""
import ctypes
import numpy.ctypeslib as ctl 
import numpy as np
import os
libfile = os.path.dirname(__file__) + '/km_dict_lib.so'
lib = ctypes.cdll.LoadLibrary(libfile)

class KMDict:
    def __init__(self, patch_size = 15, branching_factor = 5, number_layers = 5, 
                 normalization = False):
        self.patch_size = patch_size
        self.branching_factor = branching_factor
        self.number_layers = number_layers
        self.normalization = normalization
        self.tree = None
    
    
    def build_tree(self, image, number_training_patches=30000):
        '''
        Builds a k-means search tree from image patches. Image patches of size 
        patch_size are extacted, and the tree is build by hierarchical k-means where 
        k is the branching_factor. To reduce processing time, the tree is build from 
        number_training_patches. If this exceeds the the total number of patches in 
        the image, then the trainin is done on all possible patches from the image.
        The resulting kmeans tree is a 2D array.
    
        Parameters
        ----------
        image : numpy array
            2D image with variable number of channels.
        patch_size : integer
            Side length of patch.
        branching_factor : integer
            Branching of kmtree.
        number_training_patches : integer
            Number of patches used for training the kmtree. If the number exceeds 
            the total number of patches in the image, then the trainin is done on 
            all possible patches from the image.
        number_layers : integer
            Number of layeres in the kmtree.
        normalization : Boolean, optional
            Normalize patches to unit length. The default is False.
    
        Returns
        -------
        tree : numpy array
            kmtree which is a 2D array. Each row is the average patch intensities 
            for the node. The number of column elements are patch_size x patch_size x
            channels in the image.
    
        '''
        # Check patch size input
        if ( self.patch_size < 0 or self.patch_size%2 != 1 ):
            print('Patch size must be positive and odd!')
            return -1
        
        
        # python function for building km_tree
        py_km_tree = lib.build_km_tree
        # say which inputs the function expects
        py_km_tree.argtypes = [ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_bool,
                                 ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
        # say which output the function gives
        py_km_tree.restype = None
        
        rows, cols = image.shape[0:2]
        channels = 1
        if ( image.ndim > 2 ):
            channels, rows, cols = image.shape
        
        total_patches = (rows-self.patch_size+1)*(cols-self.patch_size+1)
        if (number_training_patches > total_patches ):
            number_training_patches = total_patches
        print(f'number of training patches {number_training_patches}')
        
        # number of elements in tree
        n = int((self.branching_factor**(self.number_layers+1)-self.branching_factor)/(self.branching_factor-1))
        while( n > number_training_patches ):
            self.number_layers -= 1
            n = int((self.branching_factor**(self.number_layers+1)-self.branching_factor)/(self.branching_factor-1))
            print(f'number of layers {self.number_layers} number of elements {n}')
            
        # make input
        self.tree = np.empty((n, self.patch_size*self.patch_size*channels), dtype=np.float) # will be overwritten
        image = np.asarray(image, order='C')
        
        py_km_tree(image, rows, cols, channels, self.patch_size, self.number_layers, self.branching_factor, 
                   number_training_patches, self.normalization, self.tree)
          
    
    def search_tree(self, image):
        if ( self.tree is None):
            print('Tree si not build. Run build_tree first.')
            return None
        
        '''
        Search kmtree for all patches in the image to create an assignment image. 
        The assignment image is an image of indicies of closest kmtree node for 
        each pixel.
        
        Parameters
        ----------
        image : numpy array
            2D image.
        tree : numpy array
            kmtree which is a 2D array. Each row is the average patch intensities 
            for the node.
           DESCRIPTION.
        branching_factor : integer
            Branching of kmtree.
        normalization : Boolean, optional
            Normalize patches to unit length. The default is False.
    
        Returns
        -------
        A : numpy array
            Assignment image of the same rows and cols as the input image.
        number_nodes : integer
            Number of nodes in kmtree.
    
        '''
        # say where to look for the function
        py_search_km_tree = lib.search_km_tree
        # say which inputs the function expects
        py_search_km_tree.argtypes = [ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_bool,
                                 ctl.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")]
        # say which output the function gives
        py_search_km_tree.restype = None
        
        rows, cols = image.shape[0:2]
        channels = 1
        if ( image.ndim > 2 ):
            channels, rows, cols = image.shape
        
        number_nodes = self.tree.shape[0]
        image = np.asarray(image, order='C')
        
        # make input
        A = np.empty((rows,cols), dtype=np.int32) # will be overwritten
    
        py_search_km_tree(image, rows, cols, channels, self.tree, self.patch_size, number_nodes, self.branching_factor, self.normalization, A)
        
        return A        
        

class Dict:
    def __init__(self, n_tree_nodes, patch_size = 15):
        self.patch_size = patch_size
        self.n_tree_nodes = n_tree_nodes
        self.D = None
    

    def improb_to_dictprob(self, A, P):
        '''
        Taking image probabilities and assigning them to dictionary probabilities 
        according to an assignment image A.
    
        Parameters
        ----------
        A : numpy array
            Assignment image with the same rows and cols as the input image.
        P : numpy array
            Probability image with the same rows and cols as the image and channels
            the same as number of labels in the iamge.
        number_nodes : integer
            Number of nodes in kmtree.
        patch_size : integer
            Side length of patch. Patch size is used for determining number of labels.
    
        Returns
        -------
        D : numpy array
            Dictionary probabilities. Each row corresponds to the kmtree node and 
            the cols encodes the pixelwise probability. There are patch_size x 
            patch_size x number of labels.
    
        '''
        # say where to look for the function
        py_prob_to_dict = lib.prob_im_to_dict
        # say which inputs the function expects
        py_prob_to_dict.argtypes = [ctl.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                 ctypes.c_int, ctypes.c_int,
                                 ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
        # say which output the function gives
        py_prob_to_dict.restype = None 
    
        Pt = np.array(P.transpose((2,0,1)), order='C')
        
        rows, cols = A.shape
        number_layers = P.shape[2]
        
        # make input
        self.D = np.empty((self.n_tree_nodes, number_layers*self.patch_size**2), dtype=np.float) # will be overwritten
    
        py_prob_to_dict(A, rows, cols, Pt, number_layers, self.patch_size, self.n_tree_nodes, self.D)
    
    
    def dictprob_to_improb(self, A):
        if ( self.D is None ):
            print('Dictionary is not computed')
            return None
        '''
        Dictionary probabilities to iamge probabilities.
    
        Parameters
        ----------
        A : numpy array
            Assignment image.
        D : numpy array
            Dictionary probabilities. Each column contains patch_size x patch_size 
            x layers probabilities.
        patch_size : integer
            Side length of patch.
    
        Returns
        -------
        TYPE numpy array
            Probability image of size rows x cols x layers.
    
        '''
    
        # say where to look for the function
        py_dict_to_prob_im_opt = lib.dict_to_prob_im_opt
        # say which inputs the function expects
        py_dict_to_prob_im_opt.argtypes = [ctl.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                  ctypes.c_int, ctypes.c_int,
                                  ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                  ctypes.c_int, ctypes.c_int,
                                  ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
        # say which output the function gives
        py_dict_to_prob_im_opt.restype = None 
        
        rows, cols = A.shape
        number_layers = int(self.D.shape[1]/(self.patch_size**2))
        
    
        # make input
        P = np.empty((number_layers, rows, cols)) # will be overwritten
        
        py_dict_to_prob_im_opt(A, rows, cols, self.D, self.patch_size, number_layers, P)
        return P.transpose((1,2,0))

        
        




if __name__ == '__main__':
    
    import skimage.io
    import skimage.transform
    
    sc_fac = 1
    I_in = skimage.io.imread('../../../data/nerve_im_train.png')
    I_in = skimage.transform.rescale(I_in, sc_fac, preserve_range = True)
    image = I_in.astype(np.float)/255
    
    L_in = skimage.io.imread('../../../data/nerve_im_train_gray.png')
    r_in,c_in = L_in.shape[0:2]
    L_in = skimage.transform.rescale(L_in,sc_fac, order=0, anti_aliasing=False, preserve_range = True)
    
    r,c = L_in.shape
    l = int(np.max(L_in))
    
    label_image = np.zeros((r,c,l))
    for i in range(0,3):
        label_image[:,:,i] = (L_in == i+1).astype(float)
    
    patch_size = 15
    branching_factor = 5
    number_layers = 5
    
    # Change image shape to ensure that it works with odd shape
    image = image[:,:-11]
    label_image = label_image[:,:-11,:]
    
    # Test computing assignment image
    kmdict = KMDict(patch_size = patch_size, branching_factor = branching_factor, number_layers = number_layers)
    kmdict.build_tree(image, 10000)
    A = kmdict.search_tree(image)
    
    # Test getting dictionary probabilities
    dct = Dict(kmdict.tree.shape[0], patch_size = patch_size)
    dct.improb_to_dictprob(A, label_image)
    P = dct.dictprob_to_improb(A)
    print(P.shape)
    
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(A, cmap='jet')
    ax[1].imshow(P)
    
    S = np.argmax(P, axis=2)
    
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].imshow(L_in)
    ax[1].imshow(S)
    plt.show()
