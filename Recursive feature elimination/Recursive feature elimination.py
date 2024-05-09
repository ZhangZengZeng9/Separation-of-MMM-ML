from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Read data set
read_file1 = pd.read_excel("train-P.xlsx")
read_file1.to_csv("train-P.csv",
                 index=None,
                 header=True)
df1 = pd.DataFrame(pd.read_csv("train-P.csv"))
data = pd.read_csv(r"train-P.csv")
X = data.iloc[:, 1:]
Y = data.iloc[:, 0]

scaler = StandardScaler()
X_train = scaler.fit_transform(X)

RFC_ = RandomForestRegressor()

# RFECV to obtain the optimal number of features
selector = RFECV(RFC_, step=1, cv=5)
selector = selector.fit(X, Y)
optimal_feature_count = selector.n_features_
print("The optimal number of features：", optimal_feature_count)

# Computational feature importance
RFC_.fit(X, Y)
importances = RFC_.feature_importances_

print("The importance of all features：")
for i, importance in enumerate(importances):
    print(f"The importance of the feature{i+1}：{importance}")

# Determine the number of features
selector1 = RFE(RFC_, n_features_to_select=5, step=1)
selector1 = selector1.fit(X, Y)
X_wrapper1 = selector1.transform(X)
print("Optimal feature：")
for i, support in enumerate(selector1.support_):
    if support:
        print(f"feature{i+1}")

score = cross_val_score(RFC_, X_wrapper1, Y, cv=5).mean()
print("score：", score)

# Result saving
results = pd.DataFrame({
   'Feature Importance': importances,
   'Selected': selector1.support_
 })
results.to_excel('results.xlsx', index=False)