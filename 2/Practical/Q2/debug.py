import numpy as np
import numpy.typing as npt

def fill_cross(a: npt.NDArray[np.int8], row: int, col: int, newval: int):
    # From https://stackoverflow.com/a/56488480/4213397
    # I have no idea what the fuck this is
    n = len(a)
    if row + col >= n:
        anti_diag_start = (row+col-n+1,n-1)
    else:
        anti_diag_start = (0,row+col)
    if row > col:
        diag_start = (row-col,0)
    else:
        diag_start = (0,col-row)
    r, c = [np.ravel_multi_index(i,a.shape) for i in [diag_start, anti_diag_start]]
    a.ravel()[r:r+(n-diag_start[0]-diag_start[1])*(n+1):n+1] = newval
    a.ravel()[c:c*(n+1):n-1] = newval
    return a

test_array = np.zeros((10,10), dtype=np.int8)
test_array[4,7] = 1
print(test_array)
fill_cross(test_array, 4, 7, 1)
print(test_array)