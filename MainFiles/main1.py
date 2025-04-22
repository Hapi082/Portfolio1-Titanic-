import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#2.データの可視化

#データの読み込み
train = pd.read_csv("Data/train.csv")
test = pd.read_csv("Data/test.csv")
#学習データを特徴量と目的変数に分けておくと便利です。
train_x = train.drop(["Survived"],axis=1)
train_y = train["Survived"]

#読み込んだ学習データの情報を可視化してみます。
print(train.info())
"""  
 0   PassengerId  891 non-null    int64
 1   Survived     891 non-null    int64
 2   Pclass       891 non-null    int64
 3   Name         891 non-null    object
 4   Sex          891 non-null    object
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64
 7   Parch        891 non-null    int64
 8   Ticket       891 non-null    object
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object
 11  Embarked     889 non-null    object 
 """
#info()によって各列のデータ型やデータ欠損数が分かります。
#棒グラフにして各々の特徴量の重要性を調べていきます。
for i in ["Pclass","Sex","Age","SibSp","Parch","Cabin","Embarked"]:
    sns.barplot(x=train[i],y=train_y)
    plt.show()
#グラフから"Pclass,Sex,Embarked"列が特に重要であることが分かります。
