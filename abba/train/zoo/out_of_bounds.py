import os, sys
import numpy as np
import jpegio as jio


def get_DCT_bounds():
    bounds = {}
    [c, r] = np.meshgrid(range(8), range(8))
    C = np.zeros([8,8,8,8])
    for i in range(8):
        for j in range(8):
            for k in range(8):
                for l in range(8):
                    C[i,j,k,l] = np.cos((2*i+1)*k*np.pi/16)*np.cos((2*j+1)*l*np.pi/16)
                    
    Dp = 255*(C>0) - 128
    Dm = 255*(C<0) - 128
    M = np.zeros([8,8])
    m = np.zeros([8,8])
    for k in range(8):
        for l in range(8):
            M[k,l] = np.sum(C[:,:,k,l]*Dp[:,:,k,l])
            m[k,l] = np.sum(C[:,:,k,l]*Dm[:,:,k,l])
    M /= 4
    m /= 4
    M[0,:] /= np.sqrt(2)
    M[:,0] /= np.sqrt(2)
    m[0,:] /= np.sqrt(2)
    m[:,0] /= np.sqrt(2)
    bounds['max'] = M
    bounds['min'] = m
    return bounds

def is_outlier(name, folder, M, m):
    jpg = jio.read(os.path.join(folder, name))
    for c in range(jpg.image_components):
        QT = jpg.quant_tables[jpg.comp_info[c].ac_tbl_no]
        for k in range(8):
            for l in range(8):
                coeffs = jpg.coef_arrays[c][k:-1:8,l:-1:8]
                T = np.round(M[k,l]/QT[k,l])
                t = np.round(m[k,l]/QT[k,l])
                if (coeffs>T).any():
                    return name
                elif (coeffs<t).any():
                    return name 
                
