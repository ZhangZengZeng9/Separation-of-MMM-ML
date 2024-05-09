import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV, RFE
from sklearn.preprocessing import StandardScaler

# Read the training data set
read_file1 = pd.read_excel("train-S.xlsx")
read_file1.to_csv("train-S.csv", index=None, header=True)
df1 = pd.DataFrame(pd.read_csv("train-S.csv"))
data = pd.read_csv(r"train-S.csv")
X = data.iloc[:, 1:]
Y = data.iloc[:, 0]

# Standardized treatment
scaler = StandardScaler()
X_train = scaler.fit_transform(X)

RFC_ = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=2, max_features='sqrt')

selector = RFECV(RFC_, step=1, cv=10)
selector = selector.fit(X, Y)
optimal_feature_count = selector.n_features_

# Standardized treatment
RFC_.fit(X, Y)
importances = RFC_.feature_importances_

selector1 = RFE(RFC_, n_features_to_select=3, step=1)
selector1 = selector1.fit(X, Y)
X_wrapper1 = selector1.transform(X)

score = cross_val_score(RFC_, X_wrapper1, Y, cv=10).mean()

# Result saving
results = pd.DataFrame({
   'Feature Importance': importances,
   'Selected': selector1.support_
 })
results.to_excel('results.xlsx', index=False)

explainer = shap.Explainer(RFC_)

# Calculate the SHAP value
shap_values = explainer.shap_values(X)

df_shap_values = pd.DataFrame(shap_values, columns=X.columns)

# Save the SHAP value
df_shap_values.to_excel('shap_values.xlsx', index=False)

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['axes.labelweight'] = 'bold'

# Create a  figure
fig, ax = plt.subplots(figsize=(25, 10))
features = df_shap_values.columns.tolist()

# Draw SHAP
shap.summary_plot(shap_values, X, feature_names=features, show=False)

# Result saving
ax.set_xlabel("SHAP Value", fontsize=18)
ax.tick_params(axis='y', labelsize=15)
ax.tick_params(axis='x', labelsize=12)
plt.savefig('shap_plot_S0.png', dpi=300, bbox_inches='tight')