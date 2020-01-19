# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.datasets import load_boston

boston=load_boston()
boston_pd=pd.DataFrame(boston.data, columns=boston.feature_names)

plt.figure()
plt.hist(boston.target)
plt.title("Хистограма на цените на недв. имущество в Бостън")
plt.xlabel("Цена на кв. фут(???) в хил. долари")
plt.ylabel("Брой имоти")
plt.show()


plt.figure()
plt.title("Корелационен плот")
plt.xlabel("Брой стаи")
plt.ylabel("Цена в хил. долари")
plt.scatter(boston.data[:,5], boston.target)



plt.figure()
plt.title("Корелационен плот")
plt.xlabel("Престъпност")
plt.ylabel("Цена в хил. долари")
plt.scatter(boston.data[:,0], boston.target)

plt.figure(figsize=(8,6))
cor_matr = boston_pd.corr().round(1)
sb.heatmap(data=cor_matr, annot=True)
plt.show()



