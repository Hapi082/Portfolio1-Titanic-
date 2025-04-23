import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


#3.特徴量エンジニアリング

#データの読み込み
train = pd.read_csv("Data/train.csv")
test = pd.read_csv("Data/test.csv")
Id = test["PassengerId"]
train_x = train.drop(["Survived"],axis=1)
train_y = train["Survived"]
concat_data = pd.concat([train_x,test],sort=False)

#新たな特徴量を作成
concat_data["Family"] = concat_data["SibSp"] + concat_data["Parch"] + 1

#不要な列を削除
drop_features = ["PassengerId","Name","SibSp","Parch","Ticket","Fare","Cabin"]
concat_data = concat_data.drop(drop_features,axis=1)
#欠損値を処理
concat_data["Age"].fillna(concat_data["Age"].median(),inplace=True)
concat_data["Embarked"].fillna(concat_data["Embarked"].mode()[0],inplace=True)
#One-Hot Encoding
concat_data = pd.get_dummies(concat_data, columns=['Embarked', 'Sex'], drop_first=True)

#データを分割
train_x = concat_data.iloc[:len(train)]
test = concat_data.iloc[len(train):]

#データの確認
print(train_x.info())

#4.モデルの作成
#学習データとバリデーションデータを作成
X_train, X_val, y_train, y_val = train_test_split(
    train_x, train_y, test_size=0.2, random_state=42
)

#4.モデルの作成
model = XGBClassifier(
    learning_rate = 0.05,
    max_depth = 3,
    n_estimators = 200,
    use_label_encoder=False, # 警告を抑えるため
    eval_metric='logloss',  # 評価指標
    random_state=42
)

""" param_grid = {
    'max_depth': [3, 5, 7,9],
    'learning_rate': [0.01, 0.03, 0.05],
    'n_estimators': [100,150, 200]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy') """
#結果として、learning_rate = 0.05,max_depth = 3,n_estimators = 200,がベスト

#5.モデルの学習
model.fit(X_train,y_train)



#6.モデルの評価
y_pred_val = model.predict(X_val)#推測
val_acc = accuracy_score(y_val, y_pred_val)#評価
print("Validation Accuracy:", val_acc)

y_test_pred = model.predict(test)

# 提出用ファイルを作成
submission = pd.DataFrame({
    'PassengerId': Id,
    'Survived': y_test_pred
})

submission.to_csv("submission.csv",index=False)