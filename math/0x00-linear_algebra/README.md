# Linear Algebra

### [0. Slice Me Up](./0-slice_me_up.py)
Slicing an array

### [1. Trim Me Down](./1-trim_me_down.py)
Obtaining the middle columns of a matrix

### [2. Size Me Please](./2-size_me_please.py)
Write a function `def matrix_shape(matrix):` that calculates the shape of a matrix:
* You can assume all elements in the same dimension are of the same type/shape
* The shape should be returned as a list of integers

### [3. Flip Me Over](./3-flip_me_over.py)
Write a function `def matrix_transpose(matrix):` that returns the transpose of a 2D matrix, `matrix`:
* You must return a new matrix
* You can assume that `matrix` is never empty
* You can assume all elements in the same dimension are of the same type/shape

### [4. Line Up](./4-line_up.py)
Write a function `def add_arrays(arr1, arr2):` that adds two arrays element-wise:
* You can assume that `arr1` and `arr2` are lists of ints/floats
* You must return a new list
* If `arr1` and `arr2` are not the same shape, return `None`

### [5. Across The Planes](./5-across_the_planes.py)
Write a function `def add_matrices2D(mat1, mat2):` that adds two matrices element-wise:
* You can assume that `mat1` and `mat2` are 2D matrices containing ints/floats
* You can assume all elements in the same dimension are of the same type/shape
* You must return a new matrix
* If `mat1` and `mat2` are not the same shape, return `None`

### [6. Howdy Partner](./6-howdy_partner.py)
Write a function `def cat_arrays(arr1, arr2):` that concatenates two arrays:
* You can assume that `arr1` and `arr2` are lists of ints/floats
* You must return a new list

### [7. Gettin' Cozy](./7-gettin_cozy.py)
Write a function `def cat_matrices2D(mat1, mat2, axis=0):` that concatenates two matrices along a specific axis:
* You can assume that `mat1` and `mat2` are 2D matrices containing ints/floats
* You can assume all elements in the same dimension are of the same type/shape
* You must return a new matrix
* If the two matrices cannot be concatenated, return `None`

### [8. Ridin’ Bareback](./8-ridin_bareback.py)
Write a function `def mat_mul(mat1, mat2):` that performs matrix multiplication:
* You can assume that `mat1` and `mat2` are 2D matrices containing ints/floats
* You can assume all elements in the same dimension are of the same type/shape
* You must return a new matrix
* If the two matrices cannot be multiplied, return `None`

### [9. Let The Butcher Slice It](./9-let_the_butcher_slice_it.py)
Slicing a numpy array

### [10. I’ll Use My Scale](./10-ill_use_my_scale.py)
Write a function `def np_shape(matrix):` that calculates the shape of a `numpy.ndarray`:
* You are not allowed to use any loops or conditional statements
* You are not allowed to use `try/except` statements
* The shape should be returned as a tuple of integers

### [11. The Western Exchange](./11-the_western_exchange.py)
Write a function `def np_transpose(matrix):` that transposes `matrix`:
* You can assume that `matrix` can be interpreted as a `numpy.ndarray`
* You are not allowed to use any loops or conditional statements
* You must return a new `numpy.ndarray`

### [12. Bracing The Elements](./12-bracin_the_elements.py)
Write a function `def np_elementwise(mat1, mat2):` that performs element-wise addition, subtraction, multiplication, and division:
* You can assume that `mat1` and `mat2` can be interpreted as `numpy.ndarray`s
* You should return a tuple containing the element-wise sum, difference, product, and quotient,     respectively
* You are not allowed to use any loops or conditional statements
* You can assume that `mat1` and `mat2` are never empty

### [13. Cat's Got Your Tongue](./13-cats_got_your_tongue.py)
Write a function `def np_cat(mat1, mat2, axis=0)` that concatenates two matrices along a specific axis:
* You can assume that `mat1` and `mat2` can be interpreted as `numpy.ndarray`s
* You must return a new `numpy.ndarray`
* You are not allowed to use any loops or conditional statements
* You may use: `import numpy as np`
* You can assume that `mat1` and `mat2` are never empty

### [14. Saddle Up](./14-saddle_up.py)
Write a function `def np_matmul(mat1, mat2):` that performs matrix multiplication:
* You can assume that `mat1` and `mat2` are `numpy.ndarray`s
* You are not allowed to use any loops or conditional statements
* You may use: `import numpy as np`
* You can assume that `mat1` and `mat2` are never empty

### [15. Slice Like A Ninja ](./100-slice_like_a_ninja.py)
Write a function `def np_slice(matrix, axes={}):` that slices a matrix along a specific axes:
* You can assume that `matrix` is a `numpy.ndarray`
* You must return a new `numpy.ndarray`
* `axes` is a dictionary where the `key` is an axis to slice along and the `value` is a tuple representing the slice to make along that axis
* You can assume that axes represents a valid slice

### [16. The Whole Barn](./101-the_whole_barn.py)
Write a function `def add_matrices(mat1, mat2):` that adds two matrices:
* You can assume that `mat1` and `mat2` are matrices containing ints/floats
* You can assume all elements in the same dimension are of the same type/shape
* You must return a new matrix
* If matrices are not the same shape, return `None`
* You can assume that `mat1` and `mat2` will never be empty
