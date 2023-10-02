#cython: language_level=3str
import numpy as np
cimport numpy as np

# Declare the function with types for better performance
cpdef np.ndarray[double] double_elements_cython(np.ndarray[double] arr):
    # Create a new array to store the results
    cdef np.ndarray[double] result = np.empty_like(arr)
    
    # Loop through each element of the input array and double its value
    cdef int i
    for i in range(len(arr)):
        result[i] = arr[i] * 2
        
    return result

def group_particles_descending(np.ndarray[double] rho, np.ndarray[int, ndim=2] inn, double min_pers_ratio):
    cdef Py_ssize_t n = rho.shape[0]
    cdef np.ndarray[int] group_id = np.zeros(n, dtype=np.int32)
    cdef np.ndarray[int] group_ptr = np.arange(0, n, dtype=np.int32)
    cdef np.ndarray[double] group_rhomax = np.zeros(n)
    cdef np.ndarray[double] group_rhosad = np.zeros(n) - 1.
    cdef int ngroups = 0
    cdef np.ndarray[int] inn_denser = np.zeros(2, dtype=np.int32)
    cdef np.ndarray[Py_ssize_t] isort = np.argsort(rho)[::-1]
    cdef Py_ssize_t ii, j, k
    cdef int gida, gidb
    
    for ii in isort:
        k = 0
        for j in range(0, inn.shape[1]):
            if rho[inn[ii,j]] > rho[ii]:
                inn_denser[k] = inn[ii,j]
                k += 1
                if k == 2:
                    break
        
        if k == 0: # found no denser group
            group_id[ii] = ngroups
            group_rhomax[ngroups] = rho[ii]
            ngroups += 1
        elif k == 1:
            group_id[ii] = group_id[inn_denser[0]]
        else:
            gida = group_id[inn_denser[0]]
            gidb = group_id[inn_denser[1]]
            
            if gida == gidb:
                group_id[ii] = gida
                continue
            
            while gida != group_ptr[gida]:
                gida = group_ptr[gida]
            while gidb != group_ptr[gidb]:
                gidb = group_ptr[gidb]
            
            if gida == gidb:
                group_id[ii] = gida
                continue
            
            # Got two different groups connect
            # Could be a saddle point between the two groups
            if group_rhomax[gida] < group_rhomax[gidb]:
                igroup_low, igroup_high = gida, gidb
            else:
                igroup_low, igroup_high = gidb, gida

            if group_rhosad[igroup_low] == -1.: # Have the first (lowest) saddlepoint
                group_rhosad[igroup_low] = rho[ii]

                if group_rhomax[igroup_low] / group_rhosad[igroup_low] <= min_pers_ratio:
                    # If our group doesn't exceed the persistence threshold we merge it 
                    group_ptr[igroup_low] = igroup_high

            # Assign particle to group of densest neighbor particle
            if rho[inn_denser[0]] >= rho[inn_denser[1]]:
                group_id[ii] = group_id[inn_denser[0]]
            else:
                group_id[ii] = group_id[inn_denser[1]]

    return group_id, group_ptr[:ngroups], group_rhomax[:ngroups], group_rhosad[:ngroups]