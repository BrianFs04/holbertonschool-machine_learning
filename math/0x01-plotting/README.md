# Plotting

### [0. Line Graph](./0-line.py)
Plot `y` as a line graph:
* `y` should be plotted as a solid red line
* The x-axis should range from 0 to 10

![Line Graph](https://i.ibb.co/r4XMJwd/Line-Graph.png)

### [1. Scatter](./1-scatter.py)
Plot `x ↦ y ` as a scatter plot:
* The x-axis should be labeled `Height (in)`
* The y-axis should be labeled `Weight (lbs)`
* The title should be `Men's Height vs Weight`
* The data should be plotted as magenta points

![Scatter](https://i.ibb.co/TgH0NQS/Scatter.png)

### [2. Change of scale](./2-change_scale.py)
Plot `x ↦ y ` as a line graph:
* The x-axis should be labeled `Time (years)`
* The y-axis should be labeled `Fraction Remaining`
* The title should be `Exponential Decay of C-14`
* The y-axis should be logarithmically scaled
* The x-axis should range from 0 to 28650

![Change of scale](https://i.ibb.co/SVk3Cpf/Change-of-scale.png)


### [3. Two is better than one](./3-two.py)
Plot `x ↦ y1 ` and `x ↦ y2 ` as line graphs:
* The x-axis should be labeled `Time (years)`
* The y-axis should be labeled `Fraction Remaining`
* The title should be `Exponential Decay of Radioactive Elements`
* The x-axis should range from 0 to 20,000
* The y-axis should range from 0 to 1
* `x ↦ y1 ` should be plotted with a dashed red line
* `x ↦ y2 ` should be plotted with a solid green line
* A legend labeling `x ↦ y1 ` as `C-14 ` and `x ↦ y2 ` as `Ra-226 ` should be placed in the upper right hand corner of the plot

![Two is better than one](https://i.ibb.co/C5gwVkH/Two-better-than-one.png)


### [4. Frequency](./4-frequency.py)
Plot a histogram of student scores for a project:
* The x-axis should be labeled `Grades`
* The y-axis should be labeled `Number of Students`
* The x-axis should have bins every 10 units
* The title should be `Project A`
* The bars should be outlined in black

![Frecuency](https://i.ibb.co/cw0zG33/Frequency.png)

### [5. All in One](./5-all_in_one.py)
Plot all 5 previous graphs in one figure:
* All axis labels and plot titles should have a font size of `x-small` (to fit nicely in one figure)
* The plots should make a 3 x 2 grid
* The last plot should take up two column widths (see below)
* The title of the figure should be `All in One`

![All in One](https://i.ibb.co/ZTHfDtR/All-in-One.png)

### [6. Stacking Bars](./6-bars.py)
Plot a stacked bar graph:

* `fruit` is a matrix representing the number of fruit various people possess
  * The columns of `fruit` represent the number of fruit `Farrah`, `Fred`, and `Felicia` have, respectively
  * The rows of `fruit` represent the number of `apples`, `bananas`, `oranges`, and `peaches`, respectively
* The bars should represent the number of fruit each person possesses:
  * The bars should be grouped by person, i.e, the horizontal axis should have one labeled tick per person
  * Each fruit should be represented by a specific color:
    * `apples` = red
    * `bananas` = yellow
    * `oranges` = orange (#ff8000)
    * `peaches` = peach (#ffe5b4)
    * A legend should be used to indicate which fruit is represented by each color
  * The bars should be stacked in the same order as the rows of `fruit`, from bottom to top
  * The bars should have a width of `0.5`
* The y-axis should be labeled `Quantity of Fruit`
* The y-axis should range from 0 to 80 with ticks every 10 units
* The title should be `Number of Fruit per Person`

![Stacking bars](https://i.ibb.co/VwN75XX/Stacking-bars.png)

### [7. Gradient](./100-gradient.py)
Plot of sampled elevations on a mountain:
* The x-axis should be labeled `x coordinate (m)`
* The y-axis should be labeled `y coordinate (m)`
* The title should be `Mountain Elevation`
* A colorbar should be used to display elevation
* The colorbar should be labeled `elevation (m)`

![Gradient](https://i.ibb.co/tQ0Pq9z/Gradient.png)

### [8. PCA](./101-pca.py)
Visualize the data in 3D:
* The title of the plot should be `PCA of Iris Dataset`
* `data` is a `np.ndarray` of shape `(150, 4)`
   * `150` => the number of flowers
   * `4` => petal length, petal width, sepal length, sepal width
* `labels` is a `np.ndarray` of shape `(150,)` containing information about what species of iris each data point represents:
   * `0` => Iris Setosa
   * `1` => Iris Versicolor
   * `2` => Iris Virginica
* `pca_data` is a `np.ndarray` of shape `(150, 3)`
  * The columns of `pca_data` represent the 3 dimensions of the reduced data, i.e., x, y, and z, respectively
* The x, y, and z axes should be labeled `U1`, `U2`, and `U3`, respectively
* The data points should be colored based on their labels using the `plasma` color map

![PCA](https://i.ibb.co/6cKBfHt/PCA.png)

---
## Author
* **Brian Florez** - [BrianFs04](https://github.com/BrianFs04)

