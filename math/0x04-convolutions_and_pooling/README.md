# Convolutions and pooling

### **[0. Valid Convolution](./0-convolve_grayscale_valid.py)**
Write a function `def convolve_grayscale_valid(images, kernel):` that performs a valid convolution on grayscale images:
* `images` is a `numpy.ndarray` with shape `(m, h, w)` containing multiple grayscale images
    * `m` is the number of images
    * `h` is the height in pixels of the images
    * `w` is the width in pixels of the images
* `kernel` is a `numpy.ndarray` with shape `(kh, kw)` containing the kernel for the convolution
    * `kh` is the height of the kernel
    * `kw` is the width of the kernel
* You are only allowed to use two `for` loops; any other loops of any kind are not allowed
*  Returns: a `numpy.ndarray` containing the convolved images

**Output:**\
(50000, 28, 28)\
(50000, 26, 26)

[Valid convolution](https://i.ibb.co/tMzkBKL/Valid-Convolution.png)

---
## Author
* **Brian Florez** - [BrianFs04](https://github.com/BrianFs04)

