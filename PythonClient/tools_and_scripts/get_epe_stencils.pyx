import numpy as np
cimport numpy as np

cpdef modify_image(np.ndarray[np.uint8_t, ndim=3] img_buff, list list1, list list2):
    cdef np.ndarray[np.uint8_t, ndim=1] pixel
    cdef int i, j, k, idx
    cdef Py_ssize_t n

    for i in range(img_buff.shape[0]):
        for j in range(img_buff.shape[1]):
            pixel = img_buff[i, j]
            for k in range(len(list1)):
                n = 0
                while n < 3 and pixel[n] == list1[k][n]:
                    n += 1
                if n == 3:
                    idx = k
                    img_buff[i, j] = list2[idx]
                    break