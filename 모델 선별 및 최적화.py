
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import numpy as np

def Encoding(df, lable_list, onhot_list):
  """ 범주형데이터를 숫자형으로 변경하는 함수
  df: 변경할 데이터 프레임
  lable_list: 라벨 인코딩
  onhot_list: 원핫 인코딩
  """
  # 라벨 인코딩
  encoding_df=df.copy()
  le  = LabelEncoder()
  encoding_df[lable_list] = encoding_df[lable_list].apply(le.fit_transform)
  # 원 핫 인코딩
  encoding_df = pd.get_dummies(encoding_df, columns=onhot_list, drop_first= True, dtype=float)
  # drop_first: 첫번째 더미 삭제, dtype: 불리언에서 정수형으로변경
  print(len(encoding_df.columns))
"""예시
lable_list = ['model']
onhot_list = ['transmission', 'fuelType', 'carMake']
encoding_df = Encoding(df,lable_list,onhot_list)
encoding_df
"""
  return encoding_df

#데이터 분할
def Data_split(df, price):
  """ 데이터를 분할하는 함수
  df: 분할할 데이터 프레임
  price: 종속 변수명
  출력
  X_train: 학습데이터
  X_test: 테스트 데이터
  y_train: 학습데이터(실제값)
  y_test: 테스트데이터(실제값)
  """
  X = df.drop(columns=price)
  y = df[price]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
  print(X_train.shape, X_test.shape)
  print(y_train.shape, y_test.shape)
"""예제
X_train, X_test, y_train, y_test = Data_split(encoding_df, 'price')
"""
  return X_train, X_test, y_train, y_test

# 로그 변환
def Log_Trans(y_train, y_test):
  # y값 로그변환
  log_y_train = np.log1p(y_train)
  log_y_test = np.log1p(y_test)
  return log_y_train, log_y_test

# sd 스케일링
def Sd_Scaling(X_train, X_test, normalize_columns):
  #독립 변수 표준화(standscaler방법)
  sd_X_train = X_train.copy()
  sd_X_test = X_test.copy()
  standscaler = StandardScaler()
  #train데이터 스케일링
  sd_X_test = X_test.copy()
  sd_X_train[normalize_columns] = standscaler.fit_transform(X_train[normalize_columns])
  #test데이터 스케일링
  sd_X_test[normalize_columns] = standscaler.transform(X_test[normalize_columns])
  return sd_X_train, sd_X_test

#  표준 정규화
def Robust_Scaling(X_train, X_test, normalize_columns):
  #독립 변수 표준 정규화(RobustScaler방법)
  robust_X_train = X_train.copy()
  robust_X_test = X_test.copy()
  robustScaler = RobustScaler()
  #train데이터 스케일링
  robust_X_train[normalize_columns] = robustScaler.fit_transform(X_train[normalize_columns])
  #test데이터 스케일링
  robust_X_test[normalize_columns] = robustScaler.transform(X_test[normalize_columns])
""" 데이터 분할하고 사용
normalize_columns = ["year", "mileage", "tax", "mpg"] # 표준 정규화할 변수 리스트
log_y_train, log_y_test = Log_Trans(y_train, y_test)
sd_X_train, sd_X_test = Sd_Scaling(X_train, X_test, normalize_columns)
robust_X_train, robust_X_test = Robust_Scaling(X_train, X_test, normalize_columns)
"""
  return robust_X_train, robust_X_test

def Linear_model(X_train, X_test, y_train, y_test):
  """ 선형모델 학습 및 평가 함수
  X_train: 학습데이터
  X_test: 테스트 데이터
  y_train: 학습데이터(실제값)
  y_test: 테스트데이터(실제값)
  출력
  rg: 선형모델
  y_train_pred: 학습데이터 예측값
  y_test_pred: 테스트데이터 예측값
  """
  # 선형모델 학습
  rg = LinearRegression()
  rg.fit(X_train, y_train)

  # 예측  모델 평가
  y_train_pred = rg.predict(X_train)
  y_test_pred = rg.predict(X_test)
  # 학습 정확도
  train_accuarcy = rg.score(X_train, y_train)
  print("학습 정확도:", rg.score(X_train, y_train))
  # 이유는 valid용어를 사용하는 것이 바람직하나 분할을 test로 지정하여 용어에 혼란이 온다.
  return y_train_pred, y_test_pred, train_accuarcy

def lgbm_model(x1, x2, y1, y2):
  ### LGBMRegressor 모델 학습과 R2 결과
  X_train = x1 # X_train: 학습데이터
  X_test = x2 # X_test: 테스트 데이터
  y_train = y1 # y_train: 학습데이터(실제값)
  y_test = y2 # y_test: 테스트데이터(실제값)
  """
  출력
  rg: 선형모델
  y_train_pred: 학습데이터 예측값
  y_test_pred: 테스트데이터 예측값
  """
  # LGBMRegressor 모델 선언 후 Fitting
  lgb_model = LGBMRegressor(random_state=42) # all hyper-parameter : default
  lgb_model.fit(X_train, y_train)
    
  # Fitting된 모델로 x_valid를 통해 예측을 진행
  y_train_pred = lgb_model.predict(X_train)
  y_test_pred = lgb_model.predict(X_test)

  # 학습 정확도
  train_accuarcy = lgb_model.score(X_train, y_train)
  print("학습 정확도:",lgb_model.score(X_train, y_train))
  return y_train_pred, y_test_pred, train_accuarcy

def xgb_model(x1, x2, y1, y2):
  # LGBMRegressor 모델 학습과 R2 결과
  X_train = x1 # X_train: 학습데이터
  X_test = x2 # X_test: 테스트 데이터
  y_train = y1 # y_train: 학습데이터(실제값)
  y_test = y2 # y_test: 테스트데이터(실제값)
  """
  출력
  rg: 선형모델
  y_train_pred: 학습데이터 예측값
  y_test_pred: 테스트데이터 예측값
  """
  # XGBRegressor 모델 선언 후 Fitting
  xgb_model = XGBRegressor(random_state=42)  # all hyper-parameter : default
  xgb_model.fit(X_train, y_train)
    
  # Fitting된 모델로 예측 수행
  y_train_pred = xgb_model.predict(X_train)
  y_test_pred = xgb_model.predict(X_test)
 
  # 학습 정확도
  train_accuarcy = xgb_model.score(X_train, y_train)
  print("학습 정확도:", xgb_model.score(X_train, y_train))
  return y_train_pred, y_test_pred, train_accuarcy

def Rfr_model(x1, x2, y1, y2):
  # 랜덤 포레스트 회귀 모델 학습
  X_train = x1 # X_train: 학습데이터
  X_test = x2 # X_test: 테스트 데이터
  y_train = y1 # y_train: 학습데이터(실제값)
  y_test = y2 # y_test: 테스트데이터(실제값)
  """
  출력
  rfr: 선형모델
  y_train_pred: 학습데이터 예측값
  y_test_pred: 테스트데이터 예측값
  """
  # 랜덤 포레스트 회귀 모델 학습
  rfr = RandomForestRegressor(random_state=42)
  rfr.fit(X_train, y_train)
  # Fitting된 모델로 예측 수행
  y_train_pred = rfr.predict(X_train)
  y_test_pred = rfr.predict(X_test)
  # 랜덤 포레스트 R2-score
  # 학습 정확도
  train_accuarcy = rfr.score(X_train, y_train)
  print("학습 정확도:",rfr.score(X_train, y_train))
""" 예제
y_train_pred, y_test_pred, train_accuarcy = Linear_model(X_train, X_test, y_train, y_test)
y_train_pred, sd_y_test_pred, train_accuarcy = Linear_model(sd_X_train, sd_X_test, log_y_train, log_y_test)
y_train_pred, robust_y_test_pred, train_accuarcy = Linear_model(robust_X_train, robust_X_test, log_y_train, log_y_test)

예제
y_train_pred, y_test_pred, train_accuarcy = Rfr_model(X_train, X_test, y_train, y_test)
y_train_pred, sd_log_y_test_pred, train_accuarcy = Rfr_model(sd_X_train, sd_X_test, log_y_train, log_y_test)
y_train_pred, robust_log_y_test_pred, train_accuarcy = Rfr_model(robust_X_train, robust_X_test, log_y_train, log_y_test)
"""
  return y_train_pred, y_test_pred, train_accuarcy

def rfr_feature_importances(rfr, X_train):
  feature_names = X_train.columns
  importance_df = pd.DataFrame({
      'Feature': feature_names,
      'Importance': rfr.feature_importances_
  }).sort_values(by='Importance', ascending=False)
  print(importance_df.head())


# 역변한 필요한 경우
def Exp_y(log_y_test, log_y_test_pred):
  trans_y_test = np.expm1(log_y_test) # 실제값
  trans_y_test_pred = np.expm1(log_y_test_pred) #예상값
  return trans_y_test, trans_y_test_pred

def model_evaluation(y_test, y_test_pred, result_name ) :
  """ 모델 평가 함수
  trans_y_test: 데스트 데이터 역변환 실제값
  trans_y_test_pred: 데스트데이터 역변환 예측값
  result_name: 결과를 저장할 컬럼 이름
  """
  mse = round(mean_squared_error(y_test, y_test_pred),3) # 실제 y값, 예측값
  mae = round(mean_absolute_error(y_test, y_test_pred),3)
  rmse = round(np.sqrt(mean_squared_error(y_test, y_test_pred)),3) # 실제 y값, 예측값
  r2 = round(r2_score(y_test, y_test_pred),3)
  mape = round((mean_absolute_percentage_error(y_test, y_test_pred)*100),3)

  print(f"\nLGBM {result_name} Results\n")
  print(f"평균 절대 오차(MAE): {mae}")
  print(f"평균 제곱 오차(MSE): {rmse}")
  print(f"평균 절대비율 오차(MAPE): {mape}")
  print(f"결정 계수(R2): {r2}")

  result_list = ['mse', 'rmse', 'mae', 'mape', 'r2']
  result_name = str(result_name)
  result_df = pd.DataFrame(data=[mse, rmse, mae, mape, r2],
                           index=result_list, columns=[result_name])

""" 예제
Rfr_base = model_evaluation(y_test, y_test_pred, result_name="Rfr_base")
trans_y_test, trans_y_test_pred =  Exp_y(log_y_test, sd_log_y_test_pred)
Rfr_Log_sd = model_evaluation(trans_y_test, trans_y_test_pred, result_name="Rfr_Log+sd")
trans_y_test, trans_y_test_pred =  Exp_y(log_y_test, robust_log_y_test_pred)
Rfr_Log_robust = model_evaluation(trans_y_test, trans_y_test_pred, result_name="Rfr_Log+robust")

result_df = pd.concat([Rfr_base, Rfr_Log_sd, Rfr_Log_robust], axis=1)
result_df
"""

  return result_df

# 하이퍼 파라미터 조정 
import optuna
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

# 하이퍼 파라미터 조정 
# 데이터 로드
df = full2_data.copy()
lable_list = ['model']
onhot_list = ['transmission', 'fuelType', 'carMake']
encoding_df = Encoding(df,lable_list,onhot_list)
X = encoding_df.drop(columns='price')
y = encoding_df['price']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
normalize_columns = ["year", "mileage", "tax", "mpg"] # 표준 정규화할 변수 리스트
log_y_train, log_y_test = Log_Trans(y_train, y_test)

# 목적 함수 정의
def objective(trial):
    # 하이퍼파라미터 범위 설정
    n_estimators = trial.suggest_int("n_estimators", 50, 300) # 생성할 트리수
    max_depth = trial.suggest_int("max_depth", 3, 20) # 최대 트리 깊이
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20) # 노드 분할 최소 샘플 수
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20) # 리프의 최소 샘풀 수

    # 모델 생성
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )

    # 교차 검증을 통한 평가
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    # MSE가 낮을수록 좋은데 이를 높을 수록 좋은 의미로 변경하기 위해 음수변환을 함.
    return np.mean(score)

# Optuna 최적화 수행
study = optuna.create_study(direction="maximize")  # score 높을 수록 좋은 방향으로 모델로 최적화
study.optimize(objective, n_trials=50, n_jobs=-1) # n_trials는 Optuna가 하이퍼파라미터를 탐색하는 총 시도 횟수

# 최적 하이퍼파라미터 출력
print("Best hyperparameters:", study.best_params)

# 최적 모델 학습 및 평가
best_params = study.best_params
best_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Test MSE:", mse)
# 'n_estimators': 117, 'max_depth': 19, 'min_samples_split': 4, 'min_samples_leaf': 2

# 최종 모델 학습
# 테스트 5개정도 추출해서 예측값과 실제값 확인하기
df = full2_data.copy()
df = df[~((df['engineSize'] == 0) & (df['fuelType'].isin(["Petrol", "Diesel"])))]

# 하이퍼 파라미터
n_estimators = 126
max_depth = 20
min_samples_split = 3
min_samples_leaf = 1

# 전처리
lable_list = ['model']
onhot_list = ['transmission', 'fuelType', 'carMake']
encoding_df = Encoding(df,lable_list,onhot_list)

# 데이터 분할
X_train, X_test, y_train, y_test = Data_split(encoding_df, 'price')

# 데이터 스케일링
normalize_columns = ["mileage", "tax", "mpg"] # 표준 정규화할 변수 리스트
log_y_train, log_y_test = Log_Trans(y_train, y_test)
robust_X_train, robust_X_test = Robust_Scaling(X_train, X_test, normalize_columns)

# 모델 학습
model = RandomForestRegressor(n_estimators = 126,
                              max_depth = 20,
                              min_samples_split = 3,
                              min_samples_leaf = 1,
                              random_state=42)
model.fit(robust_X_train, log_y_train)
rfr_feature_importances(model, X_train)
train_accuarcy = model.score(robust_X_train, log_y_train)
print("학습 정확도:", model.score(robust_X_train, log_y_train))
y_test_pred = model.predict(robust_X_test)

# 예측 및 개별 트리 예측값 수집
tree_predictions = np.array([tree.predict(robust_X_test.to_numpy()) for tree in model.estimators_])
"""랜덤포레스트는 feature name을 사용하지만 결정트리에서는 사용하지 않아 오류발생
robust_X_test의 feature name을 제거하기 위해 nupy 배열로 변경"""
tree_predictions = np.expm1(tree_predictions)

# 평균 및 신뢰구간 계산
y_mean = tree_predictions.mean(axis=0)
y_std = tree_predictions.std(axis=0)
lower_bound = y_mean - 1.96 * y_std  # 95% 신뢰구간 하한
upper_bound = y_mean + 1.96 * y_std  # 95% 신뢰구간 상한


# 모델 평가
trans_y_test, trans_y_test_pred = Exp_y(log_y_test, y_test_pred)
Last_result = model_evaluation(trans_y_test, trans_y_test_pred, result_name="Last_result")
print(Last_result)

# 신뢰구간 그래프 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(range(len(trans_y_test)), trans_y_test, color='#08589e', label="Actual")
plt.scatter(range(len(trans_y_test_pred)), trans_y_test_pred, color='#2b8cbe', label="Predicted")
plt.fill_between(range(len(trans_y_test_pred)), lower_bound, upper_bound, color='#4eb3d3', alpha=0.3, label="95% CI")
plt.legend()
plt.xlabel("Sample Index")
plt.ylabel("Price")
plt.title("Predicted vs Actual with 95% Confidence Interval")
plt.show()

# 테스트 데이터 일부 샘플링
np.random.seed(42)
sample_indices = np.random.choice(len(X_test), size=10, replace=False)
X_sample = X_test.iloc[sample_indices]
y_sample = y_test.iloc[sample_indices]
# 셈플 신뢰구간 구하기
tree_predictions = np.array([tree.predict(X_sample.to_numpy()) for tree in model.estimators_])
tree_predictions = np.expm1(tree_predictions)
y_sample_mean = tree_predictions.mean(axis=0)
y_sample_std = tree_predictions.std(axis=0)
y_sample_lower_bound = y_sample_mean - 1.96 * y_sample_std
y_sample_upper_bound = y_sample_mean + 1.96 * y_sample_std

#결과 출력
result_df = X_sample.copy()
encoding_df=df.copy()
le  = LabelEncoder()
encoding_df[lable_list] = encoding_df[lable_list].apply(le.fit_transform)
result_df[lable_list] = result_df[lable_list].apply(le.inverse_transform)
for category in onhot_list:
    category_columns = [col for col in result_df.columns if col.startswith(category + "_")]
    result_df[category] = result_df[category_columns].idxmax(axis=1).str.replace(category + "_", "")
    result_df.drop(columns=category_columns, inplace=True)
result_df["Actual"] = y_sample
result_df["Predicted"] = y_sample_mean
result_df["Lower Bound (95%)"] = y_sample_lower_bound
result_df["Upper Bound (95%)"] = y_sample_upper_bound
print(result_df)

# 결과 시각화
plt.figure(figsize=(10, 5))
plt.errorbar(range(10), y_sample_mean,
             yerr=1.96 * y_sample_std, fmt='o', label="Predicted (95% CI)", color='#4eb3d3', mfc='#2b8cbe')
plt.scatter(range(10), y_sample, color='#08589e', label="Actual")
plt.xlabel("Sample Index")
plt.xticks([0,1,2,3,4,5,6,7,8,9])
plt.ylabel("Price")
plt.legend()
plt.title("Random Forest Regression: Prediction with 95% Confidence Interval")
plt.show()

result_df
