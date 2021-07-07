* [Imports and Reading the DataFrame](#imports-and-reading-the-dataframe)
* [Standardizing Variables](standardizing-variables)
* [One Hot Encoding](one-hot-encoding)
* [Pre-Requisites](pre-requisites)
* [Pre-Requisites](pre-requisites)
* [Pre-Requisites](pre-requisites)
* [Pre-Requisites](pre-requisites)







## Imports and Reading the DataFrame

Run the following lines of code to import the necessary libraries and read the dataset:

```
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sea
from kneed import KneeLocator

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D



# reading the data frame
df = pd.read_csv('https://raw.githubusercontent.com/shie-ld/datasets/main/Mall_Customers.csv')
```

Now, lets take a look at the head of the data frame:

```
df.head()
```

![](../images/ss1.png)

There are five variables in the dataset. `CustomerID` is the unique identifier of each customer in the dataset, and we can drop this variable. It doesn’t provide us with any useful cluster information.

Since gender is a categorial variable, it needs to be encoded and converted into numeric.

All other variables will be scaled to follow a normal distribution before being fed into the model. We will `standardize` these variables with a mean of `0` and a standard deviation of `1`.


## Standardizing Variables

First, lets standardize all variables in the dataset to get them around the same scale.

```
col_names = ['Annual Income (k$)', 'Age', 'Spending Score (1-100)']
features = df[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
scaled_features = pd.DataFrame(features, columns = col_names)
scaled_features.head()
```

Now, lets take a look at the head of the data frame:

![](../images/ss2.png)

We can see that all the variables have been transformed, and are now centered around zero.


## One Hot Encoding

The variable `gender` is categorical, and we need to transform this into a numeric variable.

This means that we need to substitute numbers for each category. We can do this with Pandas using `pd.get_dummies()`.

```
gender = df['Gender']
newdf = scaled_features.join(gender)

newdf = pd.get_dummies(newdf, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None)

newdf = newdf.drop(['Gender_Male'],axis=1)

newdf.head()
```

Lets take a look at the head of the data frame again:

![](../images.ss3.png)

We can see that the gender variable has been transformed. You might have noticed that we dropped `Gender_Male` from the data frame. This is because there is no need to keep the variable anymore.

The values for `Gender_Male` can be inferred from `Gender_Female,` (that is, if `Gender_Female` is `0`, then `Gender_Male` will be `1` and vice versa).
























