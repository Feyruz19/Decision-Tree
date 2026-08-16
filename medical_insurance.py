import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
df = pd.read_excel("medical_insurance_500.xlsx")

# region 1

# print(df.head(10))
# print(df.tail(10))
# print(df.shape)
# print(df.columns.tolist())
# print(df.dtypes)
# print(df.describe())
# print(df.isnull().sum())
# print(df.duplicated().sum())
# endregion

# region 2
# df['Smoker'] = df['Smoker'].map({"Yes": 1, "No": 0})
# df['ChronicDisease'] = df['ChronicDisease'].map({"Yes": 1, "No": 0})

# Q1 = df['InsuranceCost_AZN'].quantile(0.25)
# Q3 = df['InsuranceCost_AZN'].quantile(0.75)
# IQR = Q3 - Q1
# lower = Q1 - 1.5 * IQR
# upper = Q3 + 1.5 * IQR
# outliers = df[(df['InsuranceCost_AZN'] < lower) | (df['InsuranceCost_AZN'] > upper)]
# print(len(outliers))
# endregion

# region 3
# X = df[['Age', 'BMI', 'Smoker', 'ExercisePerWeek', 'ChronicDisease']]
# y = df['InsuranceCost_AZN']
# print(X.columns.tolist())
# endregion

# region 4
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print("Train:", X_train.shape[0], "Test:", X_test.shape[0])
# endregion

# region 5
# best_mae_dt = 999999
# best_params_dt = None

# criteria = ['squared_error', 'absolute_error', 'friedman_mse']
# depths = [3, 5, 7, 10]
# min_splits = [2, 4, 6]
# min_leaves = [1, 2, 4]

# for cr in criteria:
#     for d in depths:
#         for ms in min_splits:
#             for ml in min_leaves:
#                 dt = DecisionTreeRegressor(criterion=cr, max_depth=d, min_samples_split=ms, min_samples_leaf=ml, random_state=42)
#                 dt.fit(X_train, y_train)
#                 y_pred = dt.predict(X_test)
#                 mae = mean_absolute_error(y_test, y_pred)
#                 if mae < best_mae_dt:
#                     best_mae_dt = mae
#                     best_params_dt = (cr, d, ms, ml)

# print(best_params_dt)

# dt_best = DecisionTreeRegressor(criterion=best_params_dt[0], max_depth=best_params_dt[1], min_samples_split=best_params_dt[2], min_samples_leaf=best_params_dt[3], random_state=42)
# dt_best.fit(X_train, y_train)
# y_pred_dt = dt_best.predict(X_test)
# endregion

# region 6
# mae_dt = mean_absolute_error(y_test, y_pred_dt)
# mse_dt = mean_squared_error(y_test, y_pred_dt)
# rmse_dt = np.sqrt(mse_dt)
# r2_dt = r2_score(y_test, y_pred_dt)

# print("DT MAE:", mae_dt)
# print("DT MSE:", mse_dt)
# print("DT RMSE:", rmse_dt)
# print("DT R2:", r2_dt)
# endregion

# region 7
# best_mae_rf = 999999
# best_params_rf = None

# n_estimators_list = [100, 200, 300]
# max_depths = [5, 8, 12]
# min_splits = [2, 4]
# min_leaves = [1, 2]
# max_features_list = ['sqrt', 'log2']

# for n in n_estimators_list:
#     for d in max_depths:
#         for ms in min_splits:
#             for ml in min_leaves:
#                 for mf in max_features_list:
#                     rf = RandomForestRegressor(n_estimators=n, max_depth=d, min_samples_split=ms, min_samples_leaf=ml, max_features=mf, n_jobs=-1, random_state=42)
#                     rf.fit(X_train, y_train)
#                     y_pred = rf.predict(X_test)
#                     mae = mean_absolute_error(y_test, y_pred)
#                     if mae < best_mae_rf:
#                         best_mae_rf = mae
#                         best_params_rf = (n, d, ms, ml, mf)

# print("RF en yaxsi parametrler:")
# print(best_params_rf)

# rf_best = RandomForestRegressor(n_estimators=best_params_rf[0], max_depth=best_params_rf[1], min_samples_split=best_params_rf[2], min_samples_leaf=best_params_rf[3], max_features=best_params_rf[4], n_jobs=-1, random_state=42)
# rf_best.fit(X_train, y_train)
# y_pred_rf = rf_best.predict(X_test)
# endregion

# region 8
# mae_rf = mean_absolute_error(y_test, y_pred_rf)
# mse_rf = mean_squared_error(y_test, y_pred_rf)
# rmse_rf = np.sqrt(mse_rf)
# r2_rf = r2_score(y_test, y_pred_rf)

# print("RF MAE:", mae_rf)
# print("RF MSE:", mse_rf)
# print("RF RMSE:", rmse_rf)
# print("RF R2:", r2_rf)
# endregion

# region 9
# print("DT MAE:", mae_dt, "R2:", r2_dt)
# print("RF MAE:", mae_rf, "R2:", r2_rf)

# if r2_rf > r2_dt:
#     print("RF daha yaxsi")
# else:
#     print("DT daha yaxsi")

# if mae_rf < mae_dt:
#     print("RF sehv daha az")
# else:
#     print("DT sehv daha az")

# endregion

# region 10
# importance = pd.Series(rf_best.feature_importances_, index=X.columns).sort_values(ascending=False)
# print(importance)

# print(importance.head(3))

# print("En az tesir eden:")
# print(importance.index[-1], importance.iloc[-1])

# plt.figure(figsize=(10, 6))
# importance.sort_values(ascending=True).plot(kind='barh', color='teal')
# plt.title('Feature Importance')
# plt.tight_layout()
# plt.savefig('feature_importance.png', dpi=150)
# plt.close()
# endregion

# region 11
new_customer = pd.DataFrame({
    'Age': [45],
    'BMI': [31.5],
    'Smoker': [1],
    'ExercisePerWeek': [2],
    'ChronicDisease': [1]
})

print("DT proqnoz:", dt_best.predict(new_customer)[0])
print("RF proqnoz:", rf_best.predict(new_customer)[0])
# endregion
