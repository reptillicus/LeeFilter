 #boxcar filter
import numpy as np
cimport numpy as np
from scipy.ndimage.filters import uniform_filter as boxcar
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
import cython
cimport cython 

@cython.boundscheck(False)
@cython.wraparound(False)
#@cython.nonecheck(False)
def LeeFilter(np.ndarray[double, ndim=2] Array, int N = 7, float sig = 5, mode="reflect"):
    """
        N = size of filter box
        Sig = Estimate of Variance
        mode = mode for convolution, defauts to reflect
    """

    #width of the window
    cdef  int Delta = (N - 1) / 2     
    #sz = np.shape(Array)
    cdef  int n_row = Array.shape[0]    
    cdef  int n_col = Array.shape[1]
    #cdef np.ndarray[DTYPE_t, ndim=2] temp_arr
    cdef  int ntemp_row
    cdef  int ntemp_col

    cdef  float total

    cdef np.ndarray[DTYPE_t, ndim=2] mean = boxcar(Array, (N,N), mode=mode)
    cdef np.ndarray[DTYPE_t, ndim=2] z = np.zeros((n_row, n_col), dtype=DTYPE)

    cdef int nr, nc, tr, tc, v, w

    for nc in xrange(Delta, n_col - Delta):  
        for nr in xrange(Delta, n_row - Delta):
            
            # METHOD #1
            # creating the temp_arr takes a lot of overhead, ~3 sec 

            #temp_arr = Array[nr - Delta:nr + Delta, nc - Delta:nc + Delta] - mean[nr,nc]
            #ntemp_row = temp_arr.shape[0]
            #ntemp_col = temp_arr.shape[1]
            #total = 0.0
            #for tr in xrange(0, ntemp_row):
            #    for tc in xrange(0, ntemp_col):
            #        total += temp_arr[tr,tc]**2

            #z[nr, nc] = total


            # METHOD #2
            # Without types on indices v and w, this is esesntially a pure python
            # loop, and takes ~20 sec    
            total = 0.0
            for v in xrange(-Delta, Delta+1):
                for w in xrange(-Delta, Delta+1):
                    #print nc, nr, v, w, total
                    total += (Array[nr + v, nc +  w ] - mean[nr,nc])**2

            z[nr, nc] = total

            # METHOD #3
            # ~5 sec using numpy.sum()
            #z[nr, nc] = np.sum((Array[nr - Delta:nr + Delta, nc - Delta:nc + Delta] - mean[nr,nc])**2)

    z = z / (N**2 - 1)
    
    #Upon starting the next equation,  Z = Var(Z). Upon exit, Z = Var(X) 
    #of equation 19 of Lee, Optical Engineering 25(5), 636-643 (May 1986)
    #
    #VAR_X = (VAR_Z + Mean^2 )/(Sigma^2 +1) - Mean^2   (19)
    #
    #Here we constrain to >= 0, because Var(x) can't be negative:
    cdef np.ndarray[DTYPE_t, ndim=2] var_x=(z + mean**2) /(sig**2 + 1.0) - mean**2

    #return value from equation 21,22 of Lee, 1986.
    #K = ( VAR_X/(mean^2 * Sigma^2 + VAR_X) )          (22)
    #Filtered_Image = Mean + K * ( Input_Image - Mean) (21)
  
    #cdef np.ndarray[DTYPE_t, ndim=2] out_array = mean + (Array - mean) * ( var_x/(mean_squared*sig**2 + var_x) ) 

    return mean + (Array - mean) * ( var_x/(mean**2 * sig**2 + var_x) ) 

