import numpy as np
from tutorial_bs_hw_accelerator import perm_3x3

def per(mtx, column, selected, prod, output=False):
    """
    Row expansion for the permanent of matrix mtx.
    The counter column is the current column, 
    selected is a list of indices of selected rows,
    and prod accumulates the current product.
    """
    if column == mtx.shape[1]:
        return prod
    else:
        result = 0
        for row in range(mtx.shape[0]):
            if not row in selected:
                result = result \
                + per(mtx, column+1, selected+[row], prod*mtx[row,column])
        return result

def permanent(mat):
    """
    Returns the permanent of the matrix mat.
    """
    return per(mat, 0, [], 1)

def test_main():
    """
    Test on the permanent.
    """
    dim = 3 
    rmt = np.random.randint(0, 1, size=(dim, dim))
    assert np.isclose(perm_3x3(rmt), permanent(rmt))
