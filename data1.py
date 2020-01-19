# -*- coding: utf-8 -*-
"""
Пример за анализ и прогнозиране на цените на имотите в
Бостън на база 13 фактора. 

        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's


Извикайте print(boston.DESCR) за подробно описание.

Факторите се съдържат в променливата boston.datа, 
докато цените (MDEV) се съдържат в boston.target.

"""

# Импортиране на библиотеки
# Импортирането става по два начина 
# Първият е даден по-долу и импортира цялата 
# библиотека под даден неймспейс. 
# Например import numpy as np импортира библиотеката
# numpy под неймспейс np. Неймспейс np означава, 
# че всички функции, обекти и т.н. ще са достъпни
# чрез префикса np. Така например методът sort на 
# numpy ще се извика като np.sort.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# Вторият начин на импортиране е даден по-долу.
# Той импортира конкретен метод или обект 
# който се използва без префикс за неймспейс

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Ето тук load_boston се използва без никакъв префикс.
# В случая зарежда готовият тестов набор boston, 
# който се разпространява със scikit-learn

boston=load_boston()

# Долният ред създава Pandas фрейм, който е нужен за 
# визуализацията на корелационната матрица по-долу чрез 
# heatmap
boston_pd=pd.DataFrame(boston.data, columns=boston.feature_names)

# Експлораторен анализ 

# Чертане на хистограма, която дава представа за
# това колко броя жилища се намират във всеки 
# ценови диапазон
# Параметърът bins= дава колко колони трябва да 
# има хистограмата.

plt.figure()
plt.hist(boston.target, bins=30)
plt.title("Хистограма на цените на недв. имущество в Бостън")
plt.xlabel("Цена на имотите в хил. долари")
plt.ylabel("Брой имоти")
plt.show()

# Чертане на корелациионен плот, който дава връзката 
# между цената на жилището и броя стаи (който е 
# измерен на база идеални размери) и явно 
# играе еквивалентна роля на квадратния метър в България.

plt.figure()
plt.title("Корелационен плот")
plt.xlabel("Брой стаи")
plt.ylabel("Цена в хил. долари")
plt.scatter(boston.data[:,5], boston.target)

# Чертане на корелационен плот, който дава връзката 
# между престъпността в района и броя на жилищата

plt.figure()
plt.title("Корелационен плот")
plt.xlabel("Престъпност")
plt.ylabel("Цена в хил. долари")
plt.scatter(boston.data[:,0], boston.target)

# Корелационната матрица разкрива взаимната зависимост
# между отделните фактори. 
# Стойност 0 означава, че факторите са напълно независими един от 
# друг. Стойности, които се доближават до 1 означават
# силна положителна корелация (едното расте и другото расте) 
# докато стойности, които се доближават до -1 означават 
# силна отрицателна корелация (едното расте, докато другото
# намалява. 

plt.figure(figsize=(8,6))
cor_matr = boston_pd.corr().round(1)

# Корелационната матрица може да се визуализира ефектно
# като се използва heatmap от библиотеката seaborn
# annot = True изписва текста освен цветовете
ax = sb.heatmap(data=cor_matr, annot=True)

# Seaborn не работи много добре с новите версии на 
# matplotlib, което налага следната ръчна корекция.
# Без долните два реда, диаграмата ще е отрязана.
bottom, top = ax.get_ylim()
ax.set_ylim(bottom+0.5, top-0.5)
plt.show()

train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(boston.data[:,5], boston.target, test_size=0.2, random_state=5) 

train_set_x = train_set_x.reshape(-1,1)
test_set_x = test_set_x.reshape(-1,1)
train_set_y = train_set_y.reshape(-1, 1)
test_set_y = test_set_y.reshape(-1,1)

reg = LinearRegression()
reg.fit(train_set_x, train_set_y)

y_prediction = reg.predict(test_set_x)
rmse = (np.sqrt(mean_squared_error(test_set_y, y_prediction)))
print('RMSE : {}'.format(rmse))
print('R2: {}'.format(reg.score(test_set_x, test_set_y)))