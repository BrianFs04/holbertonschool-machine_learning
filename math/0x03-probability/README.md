# Probability

### **[0. Initialize Poisson](./poisson.py)**
Create a class `Poisson` that represents a poisson distribution:
* Class contructor `def __init__(self, data=None, lambtha=1.):`
    * `data` is a list of the data to be used to estimate the distribution
    * `lambtha` is the expected number of occurences in a given time frame
    * Sets the instance attribute `lambtha`
        * Saves `lambtha` as a float
    * If `data` is not given, (i.e. `None`):
        * Use the given `lambtha`
        * If `lambtha` is not a positive value, raise a `ValueError` with the message `lambtha must be a positive value`
    * If `data` is given:
        * Calculate the `lambtha` of `data`
        * If `data` is not a `list`, raise a `TypeError` with the message `data must be a list`
        * If `data` does not contain at least two data points, raise a `ValueError` with the message `data must contain multiple values`

**Output:**
Lambtha: 4.84
Lambtha: 5.0

### **[1. Poisson PMF](./poisson.py)**
Update the class `Poisson`:
* Instance method `def pmf(self, k):`
    * Calculates the value of the PMF for a given number of “successes”
    * `k` is the number of “successes”
        * If `k` is not an integer, convert it to an integer
        * If `k` is out of range, return `0`
    * Returns the PMF value for `k`

**Output:**
P(9): 0.03175849616802446
P(9): 0.036265577412911795

### **[2. Poisson CDF](./poisson.py)**
Update the class `Poisson`:
* Instance method `def cdf(self, k):`
    * Calculates the value of the CDF for a given number of “successes”
    * `k` is the number of “successes”
        * If `k` is not an integer, convert it to an integer
        * If `k` is out of range, return 0
* Returns the CDF value for `k`

**Output:**
F(9): 0.9736102067423525
F(9): 0.9681719426208609

### **[3. Initialize Exponential](./exponential.py)**
Create a class `Exponential` that represents an exponential distribution:
* Class contructor `def __init__(self, data=None, lambtha=1.):`
    * `data` is a list of the data to be used to estimate the distribution
    * `lambtha` is the expected number of occurences in a given time frame
    * Sets the instance attribute `lambtha`
        * Saves `lambtha` as a float
    * If `data` is not given, (i.e. `None`):
        * Use the given `lambtha`
        * If `lambtha` is not a positive value, raise a `ValueError` with the message `lambtha must be a positive value`
    * If `data` is given:
        * Calculate the `lambtha` of `data`
        * If `data` is not a `list`, raise a `TypeError` with the message `data must be a list`
        * If `data` does not contain at least two data points, raise a `ValueError` with the message `data must contain multiple values`

**Output:**
Lambtha: 2.1771114730906937
Lambtha: 2.0}

### **[4. Exponential PDF](./exponential.py)**
Update the class `Exponential`:
* Instance method `def pdf(self, x):`
    * Calculates the value of the PDF for a given time period
    * `x` is the time period
    * Returns the PDF value for `x`
    * If `x` is out of range, return `0`

**Output:**
f(1): 0.24681591903431568
f(1): 0.2706705664650693

### **[5. Exponential CDF](./exponential.py)**
Update the class `Exponential`:
* Instance method `def cdf(self, x):`
    * Calculates the value of the CDF for a given time period
    * `x` is the time period
    * Returns the CDF value for `x`
    * If `x` is out of range, return `0`

**Output:**
F(1): 0.886631473819791
F(1): 0.8646647167674654

### **[6. Initialize Normal](./normal.py)**
Create a class `Normal` that represents a normal distribution:
* Class contructor `def __init__(self, data=None, mean=0., stddev=1.):`
    * `data` is a list of the data to be used to estimate the distribution
    * `lambtha` is the expected number of occurences in a given time frame
    * `stddev` is the standard deviation of the distribution
    * Sets the instance attributes `mean` and `stddev`
        * Saves `mean` and `stddev` as floats
    * If `data` is not given, (i.e. `None`):
        * Use the given `mean` and `stddev`
        * If `stddev` is not a positive value, raise a `ValueError` with the message `stddev must be a positive value`
    * If `data` is given:
        * Calculate the mean and standard deviation of `data`
        * If `data` is not a `list`, raise a `TypeError` with the message `data must be a list`
        * If `data` does not contain at least two data points, raise a `ValueError` with the message `data must contain multiple values`

**Output:**
Mean: 70.59808015534485 , Stddev: 10.078822447165797
Mean: 70.0 , Stddev: 10.0

### **[7. Normalize Normal](./normal.py)**
Update the class `Normal`:
* Instance method def `z_score(self, x):`
    * Calculates the z-score of a given x-value
    * `x` is the x-value
    * Returns the z-score of `x`
* Instance method `def x_value(self, z):`
    * Calculates the x-value of a given z-score
    * `z` is the z-score
    * Returns the x-value of `z`

**Output:**
Z(90): 1.9250185174272068
X(2): 90.75572504967644

Z(90): 2.0
X(2): 90.0

### **[8. Normal PDF](./normal.py)**
Update the class `Normal`:
* Instance method `def pdf(self, x):`
    * Calculates the value of the PDF for a given x-value
    * `x` is the x-value
    * Returns the PDF value for `x`

**Output:**
PSI(90): 0.006206096804434349
PSI(90): 0.005399096651147344

### **[9. Normal CDF](./normal.py)**
Update the class `Normal`:
* Instance method `def cdf(self, x):`
    * Calculates the value of the CDF for a given x-value
    * `x` is the x-value
    * Returns the CDF value for `x`

**Output:**
PHI(90): 0.982902011086006
PHI(90): 0.9922398930667251

### **[10. Initialize Binomial](./binomial.py)**
Create a class Binomial that represents a binomial distribution:
* Class contructor `def __init__(self, data=None, n=1, p=0.5):`
    * `data` is a list of the data to be used to estimate the distribution
    * `n` is the number of Bernoulli trials
    * `p` is the probability of a “success”
    * Sets the instance attributes `n` and `p`
        * Saves `n` as an integer and `p` as a float
    * If `data` is not given (i.e. `None`)
        * Use the given `n` and `p`
        * If `n` is not a positive value, raise a `ValueError` with the message `n must be a positive value`
        * If `p` is not a valid probability, raise a `ValueError` with the message `p must be greater than 0 and less than 1`
    * If `data` is given:
        * Calculate `n` and `p` from `data`
        * Round `n` to the nearest integer
        * Hint: Calculate `p` first and then calculate `n`. Then recalculate `p`. Think about why you would want to do it this way?
        * If `data` is not a `list`, raise a `TypeError` with the message `data must be a list`
        * If `data` does not contain at least two data points, raise a `ValueError` with the message `data must contain multiple values`

**Output:**
n: 50 p: 0.606
n: 50 p: 0.6

### **[11. Binomial PMF](./binomial.py)**
Update the class `Binomial`:
* Instance method `def pmf(self, k):`
    * Calculates the value of the PMF for a given number of “successes”
    * `k` is the number of “successes”
        * If `k` is not an integer, convert it to an integer
        * If `k` is out of range, return `0`
    * Returns the PMF value for `k`

**Output:**
P(30): 0.11412829839570347
P(30): 0.114558552829524

### **[12. Binomial CDF](./binomial.py)**
Update the class `Binomial`:
* Instance method `def cdf(self, k):`
    * Calculates the value of the CDF for a given number of “successes”
    * `k` is the number of “successes”
        * If `k` is not an integer, convert it to an integer
        * If `k` is out of range, return `0`
    * Returns the CDF value for `k`
    * Hint: use the `pmf` method

**Output:**
F(30): 0.5189392017296368
F(30): 0.5535236207894576
---
## Author
* **Brian Florez** - [BrianFs04](https://github.com/BrianFs04)
