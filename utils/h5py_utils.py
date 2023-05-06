import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
import h5py
import sparse
import os

def save_sparse_matrix_to_h5py(filename,
                               data):
    '''
    Save csr or csc matrix to h5py file
    @param filename: str, path to h5py file
    @param data: csr_matrix or csc_matrix
    @return: None
    '''
    assert data.getformat() in {'csr', 'csc'}
    hf = h5py.File(filename, 'w')
    hf.create_dataset('data',
                      data=data.data)
    hf.create_dataset('indices',
                      data=data.indices)
    hf.create_dataset('indptr',
                      data=data.indptr)
    hf.create_dataset('shape',
                      data=data.shape)
    if data.getformat() == 'csr':
        hf.create_dataset('matrix_format',
                          data=[0])
    else:
        hf.create_dataset('matrix_format',
                          data=[1])
    hf.close()
    
def load_sparse_matrix_from_h5py(filename):
    '''
    Load csr or csc matrix from h5py file
    @param filename: str, path to h5py file
    @return: sparse matrix
    '''
    assert os.path.exists(filename), filename + ' does not exist'
    hf = h5py.File(filename, 'r')
    data          = hf.get('data')
    indices       = hf.get('indices')
    indptr        = hf.get('indptr')
    shape         = hf.get('shape')
    matrix_format = hf.get('matrix_format')[0]
    if matrix_format == 0:
        data = csr_matrix((data, indices, indptr), shape=shape)
    else:
        assert matrix_format == 1
        data = csc_matrix((data, indices, indptr), shape=shape)
    hf.close()
    return data

def save_coo_matrix_to_h5py(filename,
                            data,
                            binary=False):
    '''
    Save coo matrix to h5py file
    @param filename: str, path to h5py file
    @param data: sparse coo matrix
    @param binary: boolean, if True saves all non-zero entries in data as 1
    @return: None
    '''
    hf = h5py.File(filename, 'w')
    if binary:
        hf.create_dataset('binary',
                          data=[1])
    else:
        hf.create_dataset('binary',
                          data=[0])
        hf.create_dataset('data',
                          data=data.data)
    hf.create_dataset('coords',
                      data=data.coords)
    hf.create_dataset('shape',
                      data=data.shape)
    hf.close()
    
def load_coo_matrix_from_h5py(filename):
    '''
    Load coo matrix from h5py file
    @param filename: str, path to h5py file
    @return: sparse coo matrix
    '''
    assert os.path.exists(filename), filename + ' does not exist'
    hf = h5py.File(filename, 'r')
    binary   = hf.get('binary')[0]
    if binary == 1:
        data = 1
    else:
        data = hf.get('data')
    coords   = hf.get('coords')
    shape    = hf.get('shape')
    data = sparse.COO(coords,
                      data,
                      shape=shape)
    return data

def save_data_to_h5py(filename,
                      data_dict):
    '''
    Save data to h5py file
    @param filename: str, path to h5py file
    @param data_dict: dict, mapping str to numpy arrays or lists
    @return: None
    '''
    hf = h5py.File(filename, 'w')
    for data_key in data_dict.keys():
        hf.create_dataset(data_key,
                          data=data_dict[data_key])
    hf.close()

def load_data_from_h5py(filename):
    '''
    Load data from h5py file
    @param filename: str, path to h5py file
    @return: dict, mapping str to numpy arrays or lists
    '''
    assert os.path.exists(filename), filename + ' does not exist'
    hf = h5py.File(filename, 'r')
    data_dict = dict()
    for data_key in hf.keys():
        data_dict[data_key] = hf.get(data_key)[:]
    hf.close()
    return data_dict