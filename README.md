# Airbnb Toronto Price Prediction

##Introduction

Airbnb is an online marketplace that offers lodging services, homesyaing services and tourisim activities to travelors since 2008. In this project, I have performed EDA on Toronto airbnb data from june 3rd 2017. Toronto is the most populous city in Canada and is a destination to so many travallors. Hence, the Airbnb is plying a significant role
in the rental markettplace of Toronto. In this kernel, I built three classification models to choose the one with the highest prediction accuracy score. Meanwhile, I will discuss feature importance and the expolaratory analysis that I have done.
```import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import nan

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix```


First, let's explore the data content.

```
df.describe()

```



