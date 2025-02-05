"""
<Python 문서 전체 설명>
1) 전체 순서
    1. 데이터셋 준비하기
    2. 데이터셋 EDA
    3. 데이터 전처리
    4. 데이터셋 분할하기
    5. 학습 데이터를 이용한 모델 학습(모델 성능 비교)
    6. 하이퍼 파라미터 튜닝
    7. 최종 성능 측정과 평가
    8. 결과 해석
    9. 모델 저장 및 로딩 
2) #: 코드 설명, ##: 대제목, ###: 소제목
    ##과 ###은 마크다운 셀에서 실행, 일반 셀과 구분하기 위해 ***으로 감싸줌
3) 수정하기 쉽도록 문단 간의 띄어쓰기는 최종 파이썬 파일을 정리할 때 할 예정
    파이썬 파일은 코드를 전체적으로 볼 때만 사용
"""
## ***1. 데이터셋 준비하기***
# warning 제거
import warnings
warnings.filterwarnings('ignore')
# 필요 라이브러리 가져오기
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import statsmodels.api as sm
import os
# 데이터 파일 확인
input_path = os.getcwd() + '/used_car copy'  # 데이터셋이 저장된 경로
file_paths = []

for dirname, _, filenames in os.walk(input_path):
    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        if filename.endswith('.csv'):  # CSV 파일만 선택
            file_paths.append(file_path)

print(f"총 {len(file_paths)}개의 CSV 파일을 발견했습니다.")
print(file_paths)
# CSV 파일 읽기 및 병합
data_frames = []

for file_path in file_paths:
    df = pd.read_csv(file_path)
    print(f"{file_path} 파일 읽기 완료. 데이터 크기: {df.shape}")
    df["carMake"] = file_path.split('/')[-1].split('.')[0].split('\\')[1]
    data_frames.append(df)    
# 모든 데이터프레임을 하나로 병합
if data_frames:
    full_data = pd.concat(data_frames, ignore_index=True)
    print(f"병합된 데이터 크기: {full_data.shape}")
else:
    print("CSV 파일이 없습니다.")    
# 데이터프레임 확인
print("데이터의 첫 5행 미리보기:")
print(full_data.head())
# 데이터셋 원본파일(full_data) 내보내기
# full_data.to_csv('full_data.csv', index=False) # 파일 실행마다 파일 생성을 방지하기 위해 주석 처리
## ***2. 데이터셋 EDA***
# 데이터 컬럼 의미 확인
# model: 모델 구분 / year: 출시 년도 / price: 가격(Pound) / transmission: 변속기 유형(요즘 대부분 오토)
# mileage: 주행 거리 (mile) / fuelType: 연료 유형(Petrol: 휘발유, Disel: 경유) / tax: 연간 도로세 (Pound) 
# / mpg(miles per gallon): 1갤런 당 주행거리(mile) 즉 연비 / engineSize: 엔진 크기(단위:리터)
full_data.head()
### ***EDA를 위한 최소한의 전처리***
# 전처리를 위한 복사본 만들기
full2_data = full_data.copy(deep=True)

# 명목형 변수 값의 공백을 삭제
str_list = ['model', 'transmission', 'fuelType', 'carMake'] # 명목변수
for i in str_list:
    full2_data[i] = full2_data[i].str.strip()
print(full2_data['model'].unique()[:20])
# tax열 결측치 채우기: tax 와 tax(£) 컬럼 합치기
full2_data['tax'].fillna(full2_data['tax(£)'], inplace=True)
full2_data.drop(columns='tax(£)', inplace=True)
full2_data.isna().sum()
# 전체 데이터 확인
print('1. 데이터 프레임 요약:')
print(full2_data.info())
print(f'\n2. 결측값 확인: \n{full2_data.isna().sum()}')
print(f'\n3. 기술통계 요약: \n{full2_data.describe()}')
# 데이터 간 상관관계 히트맵 표현
sns.heatmap(data=full2_data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Feature correlationship heatmap')
plt.show()
# scatterplot 그리기

# Subplot 설정
num_features = len(full2_data.columns) -1  # 독립 변수 개수
cols = 3  # 한 줄에 들어갈 플롯 수
rows = (num_features + cols - 1) // cols  # 행 개수 계산

fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
fig.suptitle("Scatter Plots: price vs Features", fontsize=16)

# 각 독립 변수와 price의 산점도 그리기
for i, column in enumerate(full2_data.columns.drop('price')):
    ax = axes[i // cols, i % cols]
    sns.scatterplot(data=full2_data, x=column, y="price", ax=ax)
    ax.set_title(f"price vs {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("price")

# 빈 subplot 숨기기 (독립변수 개수가 subplot보다 적은 경우)
for j in range(i + 1, rows * cols):
    fig.delaxes(axes[j // cols, j % cols])

plt.tight_layout(rect=[0, 0, 1, 0.95])  # 제목과 간격 조정
plt.show()
# 가격별 산점도(선형성 확인)
import seaborn as sns
import matplotlib.pyplot as plt
# 이산형 'year'
# 연속형 'price','mileage','tax','tax(£)', 'engineSize'
num_colums = df._get_numeric_data().columns.tolist()
# sns.lmplot(data= df, x='year', y='price',
#     line_kws={'color':'green'}
# )
# plt.ylim(0)
# plt.show()
# sns.lmplot(data= df, x='mileage', y='price',
#     line_kws={'color':'green'}
# )
# plt.ylim(0)
# plt.show()

for i in num_colums:
  if i != 'price':
    sns.lmplot(data= df, x=i, y='price',
        line_kws={'color':'green'}
    )
    plt.ylim(0)
    plt.show()
# 다중공선성 확인
from statsmodels.stats.outliers_influence import variance_inflation_factor
num_colums = full2_data._get_numeric_data().columns.tolist()
num_df = full2_data[num_colums].drop(['price'], axis=1)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(num_df.values, i) for i in range(num_df.shape[1])] # NaN 제거 하고 해야함
vif["features"] = num_df.columns
vif = vif.sort_values(by="VIF Factor", ascending=False)
vif = vif.reset_index().drop(columns='index')
print(vif)
if vif['VIF Factor'][0] > 10:
  print("다중공선성 존재")
  print(vif[vif['VIF Factor'] > 10])
else:
  print("다중공선성 존재하지 않음")

# 수치형 feature에 대해 박스플롯 그리기

plt.figure(figsize=(20, 15))  # 전체 플롯 크기 설정
for i, column in enumerate(full2_data.columns, 1):
    plt.subplot(4, 4, i)  # 4x4 subplot 생성
    sns.boxplot(y=full2_data[column], color="skyblue")
    plt.title(column, fontsize=12)
    plt.tight_layout()  # 플롯 간격 조정

plt.show()
# 가격 분포 시각화: 1,000 ~ 5,000 만원 구간에 대부분 분포함
plt.figure(figsize=(8,6))
sns.histplot(full2_data['price'], kde=True, bins=30, color="blue")
plt.title("Price Histogram")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()
## ***4. 데이터 전처리***
### ***이상치 제거***
# 모델 별 이상치 확인: 너무 길어서 주석 처리
#unique_model = full2_data['model'].unique()
#count_nums=0
#for i in unique_model:
#    model_year = full2_data[full2_data['model'] == i]['year']
#    if any(x < 2000 for x in model_year):
#        count_nums += 1
#        year_before_2000 = [x for x in model_year if x < 2000]
#        print(f'model name:{i}, count:{len(model_year)}')
#        print(f'2000년 이전 year 수:{len(year_before_2000)}')
#        plt.boxplot(model_year)
#        plt.show()
#    elif any(x > 2020 for x in model_year):
#        count_nums += 1
#        year_after_2020 = [x for x in model_year if x > 2020]
#        print(f'model name:{i}, count:{len(model_year)}')
#        print(f'2000년 이후 year 수:{len(year_before_2000)}')
#        plt.boxplot(model_year)
#        plt.show()
#print(f'count_nums:{count_nums}')
# year별 개수 세기
full2_data_cnt =full2_data.groupby(by='year').agg('count')
full2_data_cnt
# year의 누적합과, 누적비율을 세기: 2014년~2020년은 데이터의 95%로서 적당한 시기임
full2_data_cnt['cumsum'] =full2_data_cnt['model'].cumsum()
full2_data_cnt['cumsum_ratio'] = full2_data_cnt['cumsum'] / 99187 * 100
print(full2_data_cnt[['model', 'cumsum', 'cumsum_ratio']])
sns.lineplot(data=full2_data_cnt, x='year', y='cumsum_ratio')
plt.show()
# IQR 기준으로 year 이상치 탐지
Q1 = full2_data["year"].quantile(0.25)
Q3 = full2_data["year"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = full2_data[(full2_data["year"] < lower_bound) | (full2_data["year"] > upper_bound)]
print(outliers)
# year 이상치, 2060년 데이터 1개 삭제, 데이터 수집년도보다 더 최신의 날짜는 이상치임
print(full2_data.shape)
cond1 = (full2_data['year'] > 2025)
full2_data = full2_data[~cond1]
full2_data.reset_index(drop=True, inplace=True) # 행과 함께 중간의 index 1개가 삭제됨. index를 초기화
print(full2_data.shape)
print(full2_data.shape)
full2_data = full2_data[~((full2_data['engineSize'] == 0) & (full2_data['fuelType'].isin(["Petrol", "Diesel"])))]
full2_data = full2_data[full2_data["year"]>1980]
full2_data.reset_index(drop=True, inplace=True) # 행과 함께 중간의 index 1개가 삭제됨. index를 초기화
print(full2_data.shape)
full2_data
### ***인코딩***
# 인코딩 전 원본 데이터 복사
copy_full2_data = full2_data.copy(deep=True)

# 인코딩 전 컬럼 확인
print(full2_data.select_dtypes(exclude=[int, float]).columns) # 수치형 자료가 아닌 열 4개
full2_data.head(3)
# 명목형 변수 별 고유값 개수 확인: 값 많으면 라벨, 적으면 원 핫 인코딩(모델의 학습 속도 고려)
print(full2_data[['model', 'transmission', 'fuelType', 'carMake']].nunique())
# 명목형 변수의 고유값 별 개수 확인1
#vc = full2_data['model'].value_counts()

# 모델의 정확도 향상을 위해 고유값 1개인 데이터 삭제 고려: train / test 데이터 간의 차이가 발생함
#print(f'{vc[vc == 1]} \n\n개수가 1인 모델의 수: {len(vc[vc == 1])}') 
#m_vc_1 = vc[vc == 1].index
#cond = full2_data['model'].isin(m_vc_1)
#
#print(f'\n{full2_data.shape}')
#full2_data = full2_data[~cond]
#print(full2_data.shape)
# 명목형 변수의 고유값 별 개수 확인2
print(full2_data['transmission'].value_counts())
print(full2_data['fuelType'].value_counts())
print(full2_data['carMake'].value_counts())
# 명목변수 = ['model', 'transmission', 'fuelType', 'carMake']
# 'model' 라벨 인코딩 (추후에 빈도 인코딩으로 변경 가능)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
model_encoded = le.fit_transform(full2_data['model'])
full2_data['model_encoded'] = model_encoded
full2_data.drop(columns='model', inplace=True)
print(full2_data.shape)
full2_data.head()
# 나머지 명목형 변수 원 핫 인코딩(원 핫 인코딩과 비교해서 더 나은 것 채택)
encoding_df = pd.get_dummies(full2_data, dtype=float)
print(len(encoding_df.columns))
encoding_df
# model 외 3개 명목형 변수 원 핫 인코딩: 18개 인코딩 열 생성, 기존 명목형 열 3개 제외 -> 15개 추가 열 생성
from sklearn.preprocessing import OneHotEncoder

str_list = ['transmission', 'fuelType', 'carMake']

for col in str_list:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    values = ohe.fit_transform(full2_data[[col]])
    columns = [f'{col}_{cat}' for cat in ohe.categories_[0]] # 원래컬럼명 + '_' + 리스트에서 넘파이 배열[0]: 3개 합치기
    df = pd.DataFrame(data=values, columns=columns)
    full2_data = pd.concat([full2_data, df], axis=1)
    full2_data.drop(columns=col, inplace=True)

print(full2_data.shape)
print(full2_data.columns)
full2_data

## ***4. 데이터셋 분할하기***
# 데이터 분할
from sklearn.model_selection import train_test_split

X = full2_data.drop(columns='price')
y = full2_data['price']

X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.3, random_state=42)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
### ***스케일링***
# 수치형 독립변수 표준 스케일링, 종속변수 로그변환  
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

def sd_scaling(X_train, X_test, y_train, y_test, normalize_columns):
  #종속변수 로그변환, 독립변수 표준화
  # y값 로그변환
  log_y_train = np.log1p(y_train)
  log_y_test = np.log1p(y_test)
  #독립 변수 표준화(standscaler방법)
  standscaler = StandardScaler()
  #train데이터 스케일링
  sd_X_train = X_train.copy()
  sd_X_test = X_test.copy()
  sd_X_train[normalize_columns] = standscaler.fit_transform(X_train[normalize_columns])
  #test데이터 스케일링
  sd_X_test[normalize_columns] = standscaler.transform(X_test[normalize_columns])
  return sd_X_train, sd_X_test, log_y_train, log_y_test

normalize_columns = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
sd_X_train, sd_X_test, log_y_train, log_y_test = sd_scaling(X_train, X_test, y_train, y_test, normalize_columns)
sd_X_train[normalize_columns].std()
# 수치형 독립변수 로버스트 스케일링, 종속변수 로그변환  
def robust_scaling(X_train, X_test, y_train, y_test, normalize_columns):
  #종속변수 로그변환, 독립변수 표준 정규화
  # y값 로그변환
  log_y_train = np.log1p(y_train)
  log_y_test = np.log1p(y_test)
  #독립 변수 표준 정규화(RobustScaler방법)
  robust_X_train = X_train.copy()
  robust_X_test = X_test.copy()
  robustScaler = RobustScaler()
  #train데이터 스케일링
  robust_X_train[normalize_columns] = robustScaler.fit_transform(X_train[normalize_columns])
  #test데이터 스케일링
  robust_X_test[normalize_columns] = robustScaler.transform(X_test[normalize_columns])
  return robust_X_train, robust_X_test, log_y_train, log_y_test


normalize_columns = ['year', 'mileage', 'tax', 'mpg']
robust_X_train, robust_X_test, log_y_train, log_y_test = robust_scaling(X_train, X_test, y_train, y_test, normalize_columns)
### ***로그 변환 차이 분포도로 확인***
# 독립변수 표준화한 값을 그래프로 나타내기

stand_num_df = sd_X_train.loc[:, ['year', 'mileage', 'tax', 'mpg', 'engineSize']]

from scipy import stats
from scipy.stats import norm, skew

figure = plt.figure(figsize=(20,15))
rows, cols = 2, 3

for i in range(1, 6): 
    ax = figure.add_subplot(rows, cols, i)
    
    if i <= len(stand_num_df):
        stand_column = stand_num_df.columns[i-1]
        x1 = stand_num_df[stand_column]
        
        sns.distplot(x1, fit=norm)
        (x1_mu, x1_sigma) = norm.fit(x1)
                
        ax.legend([f'Stand Scaled dist.\n(x1 $\mu=$ {x1_mu:.2f}\nx1 $\sigma=$ {x1_sigma:.2f} )'], loc='best')
        
        plt.xlabel(f'{stand_column}')
        plt.ylabel('Frequency')

    else:
        ax.axis('off')

plt.tight_layout()
plt.show()
# 숫자형 독립변수, 종속변수속변수 로그 변환했을 때 분포도 확인
log_df = full2_data.copy()
log_df[['year', 'price','mileage', 'tax', 'mpg', 'engineSize']] = np.log1p(log_df[['year', 'price','mileage', 'tax', 'mpg', 'engineSize']])
log_num_df = log_df.loc[:, ['year', 'price','mileage', 'tax', 'mpg', 'engineSize']]

print(log_num_df.head())
from scipy import stats
from scipy.stats import norm, skew

figure = plt.figure(figsize=(20,15))
rows, cols = 2, 3
        
for i in range(1, 7): 
    ax = figure.add_subplot(rows, cols, i)
    
    if i <= len(log_num_df):
        log_column = log_num_df.columns[i-1]
        x2 = log_num_df[log_column]
            
        sns.distplot(x2, fit=norm)
        (x2_mu, x2_sigma) = norm.fit(x2)
        
        ax.legend([f'Log dist.\n(x2 $\mu=$ {x2_mu:.2f}\nx2 $\sigma=$ {x2_sigma:.2f} )'], loc='best')
        
        plt.xlabel(f'{log_column}')
        plt.ylabel('Frequency')

    else:
        ax.axis('off')

plt.tight_layout()
plt.show()
## ***5. 학습 데이터를 이용한 모델 학습(모델 성능 비교)***
### ***선형 회귀 모델***
# 선형모델 학습
from sklearn.linear_model import LinearRegression
rg = LinearRegression()
rg.fit(X_train, y_train)
# 예측  모델 평가
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score

y_train_pred = rg.predict(X_train)
y_test_pred = rg.predict(X_test)
accuarcy = rg.score(X_train, y_train) # 샘플값, 실제값
print(f"학습 정확도: {accuarcy:.3f}")
accuarcy = rg.score(X_test, y_test)
mae = mean_absolute_error(y_test, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred)) # 실제 y값, 예측값
r2 = r2_score(y_test, y_test_pred) #y_true, y_pred
mape = mean_absolute_percentage_error(y_test, y_test_pred)

print(f"평균 절대 오차(MAE): {mae:.3f}")
print(f"평균 제곱 오차(MSE): {rmse:.3f}")
print(f"평균 절대비율 오차(MAPE): {mape*100:.3f}")
print(f"결정 계수(R2): {r2:.3f}")
print(f"정확도: {accuarcy:.3f}")
def get_top_error_data(y_test, pred, n_tops = 5):
    # 예측값, 실제값 데이터 프레임
    result_df = pd.DataFrame(y_test.values, columns=['real_price'])
    result_df['predicted_price']= np.round(pred)
    result_df['diff'] = np.abs(result_df['real_price'] - result_df['predicted_price'])

    # 예측값과 실제값의 차이가 큰 순서로 출력
    print(result_df.sort_values('diff', ascending=False)[:n_tops])

get_top_error_data(y_test, y_test_pred, n_tops=10)
# price값 정규성 확인
from scipy import stats

fig, axs = plt.subplots(1,2, figsize = (15,6))

sns.histplot(full2_data["price"], ax =axs[0], kde=True)
stats.probplot(full2_data["price"], dist='norm', fit=True, plot=axs[1]) # QQplot

plt.show()
# price 로그 변환 후 분포 확인: 9.5를 중심으로 정규분포
sns.histplot(np.log1p(full2_data["price"]), kde=True)
plt.show()
# 로그변환 한 데이터셋을 선형모델에 학습
from sklearn.linear_model import LinearRegression
lr_log = LinearRegression()
lr_log.fit(X_train, log_y_train)
# 예측  모델 평가
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score
y_pred_test = np.expm1(lr_log.predict(X_test))

# 학습 정확도 측정
accuarcy = lr_log.score(X_train, log_y_train)
print(f'학습 정확도: {accuarcy:.4f}')

# 모델 평가
print('lr_log 평가 지표')
print(f'MAE: {mean_absolute_error(y_test, y_pred_test):.4f}') # 평균 절대 오차
print(f'MSE: {mean_squared_error(y_test, y_pred_test):.4f}') # 평균 제곱 오차
print(f'RMSE: {root_mean_squared_error(y_test, y_pred_test):.4f}') # MSE의 제곱근
print(f'MAPE: {mean_absolute_percentage_error(y_test, y_pred_test) * 100:.2f}') # 실제 값에 비해 차이가 어느 정도인지 퍼센트로 나타냄
print(f'R2_score: {r2_score(y_test, y_pred_test):.4f}') # 실제값의 분산 대비 예측값의 분산
### ***XGBOOST 모델***
# 베이스 모델 학습
from xgboost import XGBRegressor

xgb_base=XGBRegressor(random_state=42)
xgb_base.fit(X_train, y_train)

# 베이스 모델 평가
y_pred = xgb_base.predict(X_test)
accuarcy = xgb_base.score(X_train, y_train) # 샘플값, 실제값
print(f"학습 정확도: {accuarcy:.3f}")

print(f'xgb_base 평가 지표')
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae:.4f}') # 평균 절대 오차
print(f'MSE: {mse:.4f}') # 평균 제곱 오차
print(f'RMSE: {rmse:.4f}') # MSE의 제곱근
print(f'MAPE: {mape:.2f}') # 실제 값에 비해 차이가 어느 정도인지 퍼센트로 나타냄
print(f'R2_score: {r2:.4f}') # 실제값의 분산 대비 예측값의 분산
# 독립변수 표준 스케일링
xgb_sd=XGBRegressor(random_state=42)
xgb_sd.fit(sd_X_train, y_train)

y_pred = xgb_sd.predict(sd_X_test)

print(f'xgb_base 평가 지표')
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae:.4f}') # 평균 절대 오차
print(f'MSE: {mse:.4f}') # 평균 제곱 오차
print(f'RMSE: {rmse:.4f}') # MSE의 제곱근
print(f'MAPE: {mape:.2f}') # 실제 값에 비해 차이가 어느 정도인지 퍼센트로 나타냄
print(f'R2_score: {r2:.4f}') # 실제값의 분산 대비 예측값의 분산
# 독립변수 표준 스케일링 + 종속변수 로그 변환
xgb_sd_log=XGBRegressor(random_state=42)
xgb_sd_log.fit(sd_X_train, log_y_train)

y_pred = np.expm1(xgb_sd_log.predict(sd_X_test))

print(f'xgb_sd_log 평가 지표')
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae:.4f}') # 평균 절대 오차
print(f'MSE: {mse:.4f}') # 평균 제곱 오차
print(f'RMSE: {rmse:.4f}') # MSE의 제곱근
print(f'MAPE: {mape:.2f}') # 실제 값에 비해 차이가 어느 정도인지 퍼센트로 나타냄
print(f'R2_score: {r2:.4f}') # 실제값의 분산 대비 예측값의 분산
# 독립변수 로버스트 스케일링
xgb_robust=XGBRegressor(random_state=42)
xgb_robust.fit(robust_X_train, y_train)

y_pred = xgb_robust.predict(robust_X_test)

print(f'xgb_robust 평가 지표')
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae:.4f}') # 평균 절대 오차
print(f'MSE: {mse:.4f}') # 평균 제곱 오차
print(f'RMSE: {rmse:.4f}') # MSE의 제곱근
print(f'MAPE: {mape:.2f}') # 실제 값에 비해 차이가 어느 정도인지 퍼센트로 나타냄
print(f'R2_score: {r2:.4f}') # 실제값의 분산 대비 예측값의 분산
# 독립변수 로버스트 스케일링 + 종속변수 로그 변환
xgb_robust_log=XGBRegressor(random_state=42)
xgb_robust_log.fit(robust_X_train, log_y_train)

y_pred = np.expm1(xgb_robust_log.predict(robust_X_test))

print(f'xgb_robust_log 평가 지표')
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae:.4f}') # 평균 절대 오차
print(f'MSE: {mse:.4f}') # 평균 제곱 오차
print(f'RMSE: {rmse:.4f}') # MSE의 제곱근
print(f'MAPE: {mape:.2f}') # 실제 값에 비해 차이가 어느 정도인지 퍼센트로 나타냄
print(f'R2_score: {r2:.4f}') # 실제값의 분산 대비 예측값의 분산
### ***LightGBM 모델***
# 베이스라인 모델 
import lightgbm as lgb
from lightgbm import LGBMRegressor

lgb_base = LGBMRegressor(random_state=42)
lgb_base.fit(X_train, y_train)

y_pred = lgb_base.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score

print('lgb_base 평가 지표')
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae:.4f}') # 평균 절대 오차
print(f'MSE: {mse:.4f}') # 평균 제곱 오차
print(f'RMSE: {rmse:.4f}') # MSE의 제곱근
print(f'MAPE: {mape:.2f}') # 실제 값에 비해 차이가 어느 정도인지 퍼센트로 나타냄
print(f'R2_score: {r2:.4f}') # 실제값의 분산 대비 예측값의 분산

# 결과 저장
results_regression = {}
results_regression["LightGBM lgb_base Regression"] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE":mape, "R2": r2}
# 표준화 스케일링 모델
import lightgbm as lgb

lgb_sd = LGBMRegressor(random_state=42)
lgb_sd.fit(sd_X_train, y_train)
# 모델 평가
y_pred = lgb_sd.predict(sd_X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score
print('lgb_sd 평가 지표')

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae:.4f}') # 평균 절대 오차
print(f'MSE: {mse:.4f}') # 평균 제곱 오차
print(f'RMSE: {rmse:.4f}') # MSE의 제곱근
print(f'MAPE: {mape:.2f}') # 실제 값에 비해 차이가 어느 정도인지 퍼센트로 나타냄
print(f'R2_score: {r2:.4f}') # 실제값의 분산 대비 예측값의 분산

# 결과 저장
results_regression["LightGBM lgb_sd Regression"] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE":mape, "R2": r2}
# 독립변수 표준화+ 종속변수 로그변환 스케일링 모델
import lightgbm as lgb

lgb_sd_log = LGBMRegressor(random_state=42)
lgb_sd_log.fit(sd_X_train, log_y_train)
# 모델 평가
y_pred = np.expm1(lgb_sd_log.predict(sd_X_test))

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score
print('lgb_sd_log 평가 지표')

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae:.4f}') # 평균 절대 오차
print(f'MSE: {mse:.4f}') # 평균 제곱 오차
print(f'RMSE: {rmse:.4f}') # MSE의 제곱근
print(f'MAPE: {mape:.2f}') # 실제 값에 비해 차이가 어느 정도인지 퍼센트로 나타냄
print(f'R2_score: {r2:.4f}') # 실제값의 분산 대비 예측값의 분산

# 결과 저장
results_regression["LightGBM lgb_sd_log Regression"] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE":mape, "R2": r2}
# 독립변수 로버스트 스케일링 + 종속변수 로그변환
import lightgbm as lgb

lgb_robust_log = LGBMRegressor(random_state=42)
lgb_robust_log.fit(robust_X_train, log_y_train)
# 모델 평가
y_pred = np.expm1(lgb_robust_log.predict(sd_X_test))

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score
print('lgb_robust_log 평가 지표')

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae:.4f}') # 평균 절대 오차
print(f'MSE: {mse:.4f}') # 평균 제곱 오차
print(f'RMSE: {rmse:.4f}') # MSE의 제곱근
print(f'MAPE: {mape:.2f}') # 실제 값에 비해 차이가 어느 정도인지 퍼센트로 나타냄
print(f'R2_score: {r2:.4f}') # 실제값의 분산 대비 예측값의 분산

# 결과 저장
results_regression["LightGBM lgb_robust_log Regression"] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE":mape, "R2": r2}
# 추가 스케일링 결과 비교
results_df = pd.DataFrame(results_regression)
results_df = results_df.applymap(lambda x: f'{x:.4f}')
results_df.loc['MAPE'] = results_df.loc['MAPE'].apply(lambda x: f'{float(x):.2f}')
results_df
### ***랜덤 포레스트 회귀 모델***
# 베이스라인 모델
from sklearn.ensemble import RandomForestRegressor

rf_base = RandomForestRegressor(random_state=42)
rf_base.fit(X_train, y_train)
# 모델 평가
y_pred_test = rf_base.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score
print('rf_base 평가 지표')
print(f'MAE: {mean_absolute_error(y_test, y_pred_test):.4f}') # 평균 절대 오차
print(f'MSE: {mean_squared_error(y_test, y_pred_test):.4f}') # 평균 제곱 오차
print(f'RMSE: {root_mean_squared_error(y_test, y_pred_test):.4f}') # MSE의 제곱근
print(f'MAPE: {mean_absolute_percentage_error(y_test, y_pred_test) * 100:.2f}') # 실제 값에 비해 차이가 어느 정도인지 퍼센트로 나타냄
print(f'R2_score: {r2_score(y_test, y_pred_test):.4f}') # 실제값의 분산 대비 예측값의 분산

# 표준화 스케일링 모델
from sklearn.ensemble import RandomForestRegressor

rf_sd_scale = RandomForestRegressor(random_state=42)
rf_sd_scale.fit(sd_X_train, y_train)
# 모델 평가
y_pred_test = rf_sd_scale.predict(sd_X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score
print('rf_sd_scale 평가 지표')
print(f'MAE: {mean_absolute_error(y_test, y_pred_test):.4f}') # 평균 절대 오차
print(f'MSE: {mean_squared_error(y_test, y_pred_test):.4f}') # 평균 제곱 오차
print(f'RMSE: {root_mean_squared_error(y_test, y_pred_test):.4f}') # MSE의 제곱근
print(f'MAPE: {mean_absolute_percentage_error(y_test, y_pred_test) * 100:.2f}') # 실제 값에 비해 차이가 어느 정도인지 퍼센트로 나타냄
print(f'R2_score: {r2_score(y_test, y_pred_test):.4f}') # 실제값의 분산 대비 예측값의 분산

# 표준화+로그변환 스케일링 모델
from sklearn.ensemble import RandomForestRegressor

rf_sd_log_scale = RandomForestRegressor(random_state=42)
rf_sd_log_scale.fit(sd_X_train, log_y_train)
# 모델 평가
y_pred_test = np.expm1(rf_sd_log_scale.predict(sd_X_test))

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score
print('rf_sd_log_scale 평가 지표')
print(f'MAE: {mean_absolute_error(y_test, y_pred_test):.4f}') # 평균 절대 오차
print(f'MSE: {mean_squared_error(y_test, y_pred_test):.4f}') # 평균 제곱 오차
print(f'RMSE: {root_mean_squared_error(y_test, y_pred_test):.4f}') # MSE의 제곱근
print(f'MAPE: {mean_absolute_percentage_error(y_test, y_pred_test) * 100:.2f}') # 실제 값에 비해 차이가 어느 정도인지 퍼센트로 나타냄
print(f'R2_score: {r2_score(y_test, y_pred_test):.4f}') # 실제값의 분산 대비 예측값의 분산

# 로버스트 스케일링 모델
from sklearn.ensemble import RandomForestRegressor

rf_robust_scale = RandomForestRegressor(random_state=42)
rf_robust_scale.fit(robust_X_train, y_train)
# 모델 평가
y_pred_test = rf_robust_scale.predict(robust_X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score
print('rf_robust_scale 평가 지표')
print(f'MAE: {mean_absolute_error(y_test, y_pred_test):.4f}') # 평균 절대 오차
print(f'MSE: {mean_squared_error(y_test, y_pred_test):.4f}') # 평균 제곱 오차
print(f'RMSE: {root_mean_squared_error(y_test, y_pred_test):.4f}') # MSE의 제곱근
print(f'MAPE: {mean_absolute_percentage_error(y_test, y_pred_test) * 100:.2f}') # 실제 값에 비해 차이가 어느 정도인지 퍼센트로 나타냄
print(f'R2_score: {r2_score(y_test, y_pred_test):.4f}') # 실제값의 분산 대비 예측값의 분산

# 로버스트+로그변환 스케일링 모델
from sklearn.ensemble import RandomForestRegressor

rf_robust_log_scale = RandomForestRegressor(random_state=42)
rf_robust_log_scale.fit(robust_X_train, log_y_train)
# 모델 평가: 로버스트+log가 모든 지표에서 가장 좋음
y_pred_test = np.expm1(rf_robust_log_scale.predict(robust_X_test))

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score
print('rf_robust_log_scale 평가 지표')
print(f'MAE: {mean_absolute_error(y_test, y_pred_test):.4f}') # 평균 절대 오차
print(f'MSE: {mean_squared_error(y_test, y_pred_test):.4f}') # 평균 제곱 오차
print(f'RMSE: {root_mean_squared_error(y_test, y_pred_test):.4f}') # MSE의 제곱근
print(f'MAPE: {mean_absolute_percentage_error(y_test, y_pred_test) * 100:.2f}') # 실제 값에 비해 차이가 어느 정도인지 퍼센트로 나타냄
print(f'R2_score: {r2_score(y_test, y_pred_test):.4f}') # 실제값의 분산 대비 예측값의 분산

## ***6. 하이퍼 파라미터 튜닝***
### ***Grid Search CV 활용***
## Grid Search CV를 사용하여 하이퍼파라미터 수정
#from sklearn.model_selection import GridSearchCV
#
## 랜덤 포레스트 하이퍼파라미터 설정
#rfr_params = {
#    "n_estimators": [100, 200],
#    "max_depth": [None, 10, 20],
#    "min_samples_split": [2, 5],
#    "min_samples_leaf": [1, 5, 9]}
#
## 랜덤 포레스트 모델 초기화
#rfr = RandomForestRegressor(random_state=42)
#
## GridSearchCV를 사용하여 하이퍼파라미터 튜닝
#
## Gridsearch의 scoring을 MAPE로 설정
#from sklearn.metrics import mean_absolute_percentage_error, make_scorer
#
## MAPE를 위한 사용자 정의 함수 생성
#mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
#
##GridSearchCV 실행
#
#rfr_grid = GridSearchCV(rfr, rfr_params, cv=3, scoring=mape_scorer, n_jobs=-1)
#rfr_grid.fit(robust_X_train, log_y_train)
## 최적 모델과 예측 결과 계산
#best_rfr_model = rfr_grid.best_estimator_
#y_pred_rfr = np.expm1(best_rfr_model.predict(robust_X_test))
#y_pred_rfr
## 성능 평가 지표 계산
#print(f'MAE: {mean_absolute_error(y_test, y_pred_rfr):.4f}') # 평균 절대 오차
#print(f'MSE: {mean_squared_error(y_test, y_pred_rfr):.4f}') # 평균 제곱 오차
#print(f'RMSE: {root_mean_squared_error(y_test, y_pred_rfr):.4f}') # MSE의 제곱근
#print(f'MAPE: {mean_absolute_percentage_error(y_test, y_pred_rfr) * 100:.2f}') # 실제 값에 비해 차이가 어느 정도인지 퍼센트로 나타냄
#print(f'R2_score: {r2_score(y_test, y_pred_rfr):.4f}') # 실제값의 분산 대비 예측값의 분산
#print(f'Best Params: {rfr_grid.best_params_}')
# 로버스트+로그변환 스케일링 모델에 best_params 적용하기
# best_params = {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
from sklearn.ensemble import RandomForestRegressor

rf_robust_log_scale = RandomForestRegressor(random_state=42
                                            ,max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=200)
rf_robust_log_scale.fit(robust_X_train, log_y_train)
# 모델 평가
y_pred_test = np.expm1(rf_robust_log_scale.predict(robust_X_test))

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score
print('rf_robust_log_scale 평가 지표')
print(f'MAE: {mean_absolute_error(y_test, y_pred_test):.4f}') # 평균 절대 오차
print(f'MSE: {mean_squared_error(y_test, y_pred_test):.4f}') # 평균 제곱 오차
print(f'RMSE: {root_mean_squared_error(y_test, y_pred_test):.4f}') # MSE의 제곱근
print(f'MAPE: {mean_absolute_percentage_error(y_test, y_pred_test) * 100:.2f}') # 실제 값에 비해 차이가 어느 정도인지 퍼센트로 나타냄
print(f'R2_score: {r2_score(y_test, y_pred_test):.4f}') # 실제값의 분산 대비 예측값의 분산

### ***Optuna 활용***
# optuna 설치
# pip install optuna scikit-learn
# 사용한 정의 함수
lable_list = ['model']
onhot_list = ['transmission', 'fuelType', 'carMake']

def Encoding(df, lable_list, onhot_list):
  """ 범주형데이터를 숫자형으로 변경하는 함수
  df: 변경할 데이터 프레임
  lable_list: 라벨 인코딩
  onhot_list: 원핫 인코딩
  """
  # 라벨 인코딩
  encoding_df=df.copy()
  le = LabelEncoder()
  encoding_df[lable_list] = encoding_df[lable_list].apply(le.fit_transform)
  # 원 핫 인코딩
  encoding_df = pd.get_dummies(encoding_df, columns=onhot_list, drop_first= True, dtype=float)
  # drop_first: 첫번째 더미 삭제, dtype: 불리언에서 정수형으로변경
  print(len(encoding_df.columns))
  return encoding_df

#데이터 분할
from sklearn.model_selection import train_test_split
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
  return X_train, X_test, y_train, y_test

from mmap import mmap
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

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
  sd_X_train[normalize_columns] = standscaler.fit_transform(sd_X_train[normalize_columns])
  #test데이터 스케일링
  sd_X_test[normalize_columns] = standscaler.transform(sd_X_test[normalize_columns])
  return sd_X_train, sd_X_test

def MinMax_Scaling(X_train, X_test, normalize_columns):
  #독립 변수 표준화(standscaler방법)
  mm_X_train = X_train.copy()
  mm_X_test = X_test.copy()
  minmaxscaler = MinMaxScaler()
  minmaxscaler.fit(X_train[normalize_columns])
  #train데이터 스케일링
  mm_X_train[normalize_columns] = minmaxscaler.transform(mm_X_train[normalize_columns])
  #test데이터 스케일링
  mm_X_test[normalize_columns] = minmaxscaler.transform(mm_X_test[normalize_columns])
  return mm_X_train, mm_X_test

#  표준 정규화
def Robust_Scaling(X_train, X_test, normalize_columns):
  #독립 변수 표준 정규화(RobustScaler방법)
  robust_X_train = X_train.copy()
  robust_X_test = X_test.copy()
  robustScaler = RobustScaler()
  #train데이터 스케일링
  robust_X_train[normalize_columns] = robustScaler.fit_transform(robust_X_train[normalize_columns])
  #test데이터 스케일링
  robust_X_test[normalize_columns] = robustScaler.transform(robust_X_test[normalize_columns])
  return robust_X_train, robust_X_test
from sklearn.ensemble import RandomForestRegressor

def Rfr_model(x1, x2, y1, y2):
  # 랜덤 포레스트 회귀 모델 학습
  X_train = x1 # X_train: 학습데이터
  X_test = x2 # X_test: 테스트 데이터
  y_train = y1 # y_train: 학습데이터(실제값)
  y_test = y2 # y_test: 테스트데이터(실제값)
  """
  출력
  rfr: 랜덤 포레스트 회귀 모델
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
  return rfr, y_train_pred, y_test_pred, train_accuarcy

def rfr_feature_importances(rfr, X_train):
  feature_names = X_train.columns
  importance_df = pd.DataFrame({
      'Feature': feature_names,
      'Importance': rfr.feature_importances_
  }).sort_values(by='Importance', ascending=False)
  print(importance_df.head())

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import numpy as np

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

  print(f"\nLGBM {result_name} Results")
  print(f"평균 제곱 오차(MSE): {mse}")
  print(f"평균 절대 오차(MAE): {mae}")
  print(f"평균 제곱 오차(MSE): {rmse}")
  print(f"평균 절대비율 오차(MAPE): {mape}")
  print(f"결정 계수(R2): {r2}\n")

  result_list = ['mse', 'rmse', 'mae', 'mape', 'r2']
  result_name = str(result_name)
  result_df = pd.DataFrame(data=[mse, rmse, mae, mape, r2],
                           index=result_list, columns=[result_name])
  return result_df

import optuna
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

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
#Best hyperparameters: {'n_estimators': 126, 'max_depth': 20, 'min_samples_split': 3, 'min_samples_leaf': 1}
#Test MSE: 3447442.5190360746
## ***7. 최종 성능 측정과 평가***
# 모델 학습
model = RandomForestRegressor(n_estimators = 126,
                              max_depth = 20,
                              min_samples_split = 3,
                              min_samples_leaf = 1,
                              random_state=42)
model.fit(robust_X_train, log_y_train)

rfr_feature_importances(model, X_train)
train_accuarcy = model.score(robust_X_train, log_y_train)
print("학습 정확도:", model.score(robust_X_test, log_y_test))

y_pred = np.expm1(model.predict(robust_X_test))

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score
print('best_rf_model 평가 지표')
print(f'MAE: {mean_absolute_error(y_test, y_pred):.4f}') # 평균 절대 오차
print(f'MSE: {mean_squared_error(y_test, y_pred):.4f}') # 평균 제곱 오차
print(f'RMSE: {root_mean_squared_error(y_test, y_pred):.4f}') # MSE의 제곱근
print(f'MAPE: {mean_absolute_percentage_error(y_test, y_pred) * 100:.2f}') # 실제 값에 비해 차이가 어느 정도인지 퍼센트로 나타냄
print(f'R2_score: {r2_score(y_test, y_pred):.4f}') # 실제값의 분산 대비 예측값의 분산

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

print(y_sample)

# 엔진 사이즈 이상치, 연도 이상치, 연도 정규화X
df = df[~((df['engineSize'] == 0) & (df['fuelType'].isin(["Petrol", "Diesel"])))]
df = df[df["year"]>1980]
able_list = ['model']
onhot_list = ['transmission', 'fuelType', 'carMake']
encoding_df = Encoding(df,lable_list,onhot_list)
X_train, X_test, y_train, y_test = Data_split(encoding_df, 'price')

normalize_columns = ["mileage", "tax", "mpg", "engineSize"] # 표준 정규화할 변수 리스트

# 로그
log_y_train, log_y_test = Log_Trans(y_train, y_test)
log_rfr, y_train_pred, y_test_pred, train_accuarcy = Rfr_model(X_train, X_test, log_y_train, log_y_test)
rfr_feature_importances(log_rfr, X_train)
trans_y_test, trans_y_test_pred = Exp_y(log_y_test, y_test_pred)
rfr_Log = model_evaluation(trans_y_test, trans_y_test_pred, result_name="rfr_Log")

# 정규화
log_y_train, log_y_test = Log_Trans(y_train, y_test)
mm_X_train, mm_X_test = MinMax_Scaling(X_train, X_test, normalize_columns)
mm_rfr, y_train_pred, y_test_pred, train_accuarcy = Rfr_model(mm_X_train, mm_X_test, log_y_train, log_y_test)
rfr_feature_importances(mm_rfr, mm_X_train)
log_y_test, log_y_test_pred = Exp_y(log_y_test, y_test_pred)
rfr_mm = model_evaluation(trans_y_test, trans_y_test_pred, result_name="rfr_mm")

# 표준화
log_y_train, log_y_test = Log_Trans(y_train, y_test)
sd_X_train, sd_X_test = Sd_Scaling(X_train, X_test, normalize_columns)
sd_rfr, y_train_pred, y_test_pred, train_accuarcy = Rfr_model(sd_X_train, sd_X_test, log_y_train, log_y_test)
rfr_feature_importances(sd_rfr, sd_X_train)
trans_y_test, trans_y_test_pred = Exp_y(log_y_test, y_test_pred)
rfr_sd = model_evaluation(trans_y_test, trans_y_test_pred, result_name="rfr_sd")

# 표준 정규화
log_y_train, log_y_test = Log_Trans(y_train, y_test)
robust_X_train, robust_X_test = Robust_Scaling(X_train, X_test, normalize_columns)
robust_rfr, y_train_pred, y_test_pred, train_accuarcy = Rfr_model(robust_X_train, robust_X_test, log_y_train, log_y_test)
rfr_feature_importances(robust_rfr, robust_X_train)
trans_y_test, trans_y_test_pred = Exp_y(log_y_test, y_test_pred)
rfr_robust = model_evaluation(trans_y_test, trans_y_test_pred, result_name="rfr_robust")

result_df = pd.concat([rfr_base, rfr_Log, rfr_mm, rfr_sd, rfr_robust], axis=1)
pd.options.display.float_format = '{:.3f}'.format
result_df
## ***8. 결과 해석***
#결과 출력

result_df = X_sample.copy()
encoding_df=copy_full2_data.copy()
le  = LabelEncoder()
encoding_df[lable_list] = encoding_df[lable_list].apply(le.fit_transform)
result_df['model'] = le.inverse_transform(result_df['model_encoded'])


for category in onhot_list:
    category_columns = [col for col in result_df.columns if col.startswith(category + "_")]
    result_df[category] = result_df[category_columns].idxmax(axis=1).str.replace(category + "_", "")
    result_df.drop(columns=category_columns, inplace=True)
result_df["Actual"] = y_sample
result_df["Predicted"] = y_sample_mean
result_df["Lower Bound (95%)"] = y_sample_lower_bound
result_df["Upper Bound (95%)"] = y_sample_upper_bound
result_df['AE/Mean(%)'] = np.abs(result_df['Actual'] - result_df['Predicted']) / result_df['Actual'].mean() * 100 # 평균 대비 AE(절대 오차)

result_df.sort_values(by='AE/Mean(%)', ascending=False, inplace=True)

result_df.reset_index(drop=True, inplace=True)

# 결과 시각화
plt.figure(figsize=(10, 5))
plt.errorbar(range(10), result_df["Predicted"],
             yerr=1.96 * y_sample_std, fmt='o', label="Predicted (95% CI)", color='#4eb3d3', mfc='#2b8cbe')
plt.scatter(range(10), result_df["Actual"], color='#08589e', label="Actual")
plt.xlabel("Sample Index")
plt.xticks([0,1,2,3,4,5,6,7,8,9])
plt.ylabel("Price")
plt.legend()
plt.title("Random Forest Regression: Prediction with 95% Confidence Interval")
plt.show()

result_df
- 10개의 샘플 중 3개(0,1,2)는 오차율이 크지만(89, 67, 62(%)), 나머지 7개는 오차율이 매우 작게(2~32(%)) 나타나 예상한 가격을 신뢰할 수 있다.
- 오차율 계산 방법: 각각 (실제값 - 예측값)의 절댓값을 실제값의 평균으로 나눔
## ***9. 모델 저장 및 로딩***
## 모델 저장
#import joblib
#joblib.dump(model, 'best_rf_model.pkl') # 파일 실행마다 파일 생성을 방지하기 위해 주석 처리
# 불러온 모델로 예측 및 평가하기: 출력한 모델과 같은 결과
loaded_model = joblib.load('best_rf_model.pkl')

y_pred = np.expm1(loaded_model.predict(robust_X_test))

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score
print('best_rf_model 평가 지표')
print(f'MAE: {mean_absolute_error(y_test, y_pred):.4f}') # 평균 절대 오차
print(f'MSE: {mean_squared_error(y_test, y_pred):.4f}') # 평균 제곱 오차
print(f'RMSE: {root_mean_squared_error(y_test, y_pred):.4f}') # MSE의 제곱근
print(f'MAPE: {mean_absolute_percentage_error(y_test, y_pred) * 100:.2f}') # 실제 값에 비해 차이가 어느 정도인지 퍼센트로 나타냄
print(f'R2_score: {r2_score(y_test, y_pred):.4f}') # 실제값의 분산 대비 예측값의 분산
## ***11. 시각화***
### full2_data EDA
### 기초 통계량
# 시각화를 위한 원본 데이터 가져오기
full2_data = pd.read_csv('full_data.csv')
full2_data.info()
# 전처리를 위한 복사본 만들기
full2_data = full_data.copy(deep=True)

# 명목형 변수 값의 공백을 삭제
str_list = ['model', 'transmission', 'fuelType', 'carMake'] # 명목변수
for i in str_list:
    full2_data[i] = full2_data[i].str.strip()
print(full2_data['model'].unique()[:20])
# tax열 결측치 채우기: tax 와 tax(£) 컬럼 합치기
full2_data['tax'].fillna(full2_data['tax(£)'], inplace=True)
full2_data.drop(columns='tax(£)', inplace=True)
full2_data.isna().sum()
full2_data.isnull().sum()
full2_data.describe(include='all').round(4)
# full2_ data 컬럼별 시각화 만들기
num_cols = ['year', 'price', 'mileage', 'tax', 'mpg', 'engineSize']
object_cols = ['model', 'transmission', 'fuelType', 'carMake']
# 히스토그램
n, bins, patches = full2_data[num_cols].hist(bins=50,
                                              figsize=(8, 8),
                                              color = 'deepskyblue',
                                              edgecolor = 'whitesmoke',
                                              linewidth=0.5)

plt.tight_layout()
plt.show()
# 박스 플롯
num_cols_df = pd.DataFrame(full2_data.loc[:, num_cols])

figure = plt.figure(figsize=(7,10))
rows, cols = 3, 2

plt.style.use('default')

for i in range(1, 7): 
    ax = figure.add_subplot(rows, cols, i)
    
    if i <= len(num_cols_df):
        num = num_cols_df.columns[i-1]
        x1 = num_cols_df[num]
        
        plt.boxplot(x1, widths=4,
                    vert=True,
                    boxprops=dict(color='grey'),
                    medianprops=dict(color='deepskyblue'))
        
        plt.xlabel(f'{num}')
        plt.ylabel('Frequency')

    else:
        ax.axis('off')

    plt.grid(True)
    plt.tight_layout()
    
plt.show()
# 범주형 데이터
# carMake 파이차트
full2_carmake = full2_data['carMake'].value_counts()
full2_carmake = pd.DataFrame(full2_carmake)
full2_carmake.reset_index(inplace=True)
full2_carmake
full2_carmake['ratio'] = full2_carmake['count']/full2_carmake['count'].sum() * 100
full2_carmake
fig = plt.figure(figsize=(8,8))
fig.set_facecolor('white')
ax = fig.add_subplot()

colors = sns.color_palette('GnBu', 12)
wedgeprops={'width': 0.6, 'edgecolor': 'w', 'linewidth': 5}

pie = ax.pie(full2_carmake['ratio'],
              autopct='%.2f%%',
              startangle=90,
              counterclock=False,
              pctdistance=0.75,
              colors=colors,
              wedgeprops=wedgeprops,
              textprops={'size':10})

lables = full2_carmake['carMake']
for i,l in enumerate(lables):
    ang1, ang2 = pie[0][i].theta1, pie[0][i].theta2 ## 각1, 각2
    r = pie[0][i].r ## 원의 반지름

    x = ((r+1.2)/2)*np.cos(np.pi/180*((ang1+ang2)/2)) ## 정중앙 x좌표
    y = ((r+1.2)/2)*np.sin(np.pi/180*((ang1+ang2)/2)) ## 정중앙 y좌표

    ax.text(x,y,f'{lables[i]}',ha='center',va='center')

plt.show()
# model 막대 그래프
full2_model = full2_data.groupby(['carMake','model'])['model'].count().reset_index(name='count')

full2_model = pd.DataFrame(full2_model)
full2_model.reset_index(inplace=True)

full2_model.drop(axis=1, columns='index', inplace=True)
full2_model.head()
full2_model['ratio'] = round(full2_model['count']/full2_model['count'].sum()*100, 3)
full2_model.sort_values(by=['ratio'], ascending=False)
func = lambda g: g.sort_values(by='count', ascending=False)[:3]
full2_model_pivot = full2_model.groupby('carMake').apply(func)
full2_model_pivot.reset_index(level=1, drop=True, inplace=True)
full2_model_pivot.rename(columns={'carMake':'Brand'}, inplace=True)
full2_model_pivot
full2_model_pivot.reset_index(inplace=True)
full2_model_pivot
plt.figure(figsize=(20, 15))
bar = sns.barplot(data=full2_model_pivot,
                  x='carMake', y='ratio', hue='model',
                  palette='GnBu'
                  )

bar.legend(bbox_to_anchor=(1.05,1), loc='best')
plt.show()
# transmission 막대 그래프
full2_data['transmission'].unique()
full2_trans = full2_data.groupby('transmission')['transmission'].count().reset_index(name='count')
full2_trans
full2_trans['ratio'] = round(full2_trans['count']/full2_trans['count'].sum() * 100, 3)
full2_trans
from re import I
fig = plt.figure(figsize=(5,5))

colors = sns.color_palette('GnBu', 4)

bar = sns.barplot(data=full2_trans,
                  x='transmission', y='ratio',
                  width=0.5,
                  palette='GnBu'
                  )
bar.set_yticks(np.arange(0, 65, 5))

x = full2_trans['transmission']
y = full2_trans['ratio']

for i, v in enumerate(x):
    plt.text(v, y[i], y[i], # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
             fontsize = 9,
             color='gray',
             horizontalalignment='center',  # horizontalalignment (left, center, right)
             verticalalignment='bottom')    # verticalalignment (top, center, bottom)

bar.grid(axis='y', color='gray', alpha=0.6, linestyle=':')
bar.get_legend()

plt.show()
# fuelType 막대 그래프
full2_data['fuelType'].unique()
full2_fuel = full2_data.groupby('fuelType')['fuelType'].count().reset_index(name='count')
full2_fuel
full2_fuel['ratio'] = round(full2_fuel['count']/full2_fuel['count'].sum() * 100, 3)
full2_fuel
from re import I
fig = plt.figure(figsize=(5,5))

colors = sns.color_palette('GnBu', 6)

bar = sns.barplot(data=full2_fuel,
                  x='fuelType', y='ratio',
                  width=0.5,
                  palette='GnBu'
                  )
bar.set_yticks(np.arange(0, 65, 5))

x = full2_fuel['fuelType']
y = full2_fuel['ratio']

for i, v in enumerate(x):
    plt.text(v, y[i], y[i], # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
             fontsize = 9,
             color='gray',
             horizontalalignment='center',  # horizontalalignment (left, center, right)
             verticalalignment='bottom')    # verticalalignment (top, center, bottom)

bar.grid(axis='y', color='gray', alpha=0.6, linestyle=':')
bar.get_legend()

plt.show()
# scatter plot
sns.pairplot(full2_data, diag_kind='kde')
full_corr_num = full2_data.drop(columns = {'model', 'transmission', 'fuelType', 'carMake',}, axis=1)
full_corr_num.head()
full_corr_num_df = full_corr_num.corr()
full_corr_num_df
# 히트맵
plt.figure(figsize=(10,8))

sns.heatmap(data=full_corr_num_df,
            annot=True,
            fmt='.2f',
            linewidth=.3,
            cmap='GnBu')
# 모델 학습을 위한 EDA
# 독립변수 간 상관분석
# 상관관계 분석에 필요한 컬럼만 추출
full_corr = full2_data.drop(columns = {'model', 'price', 'transmission', 'fuelType', 'carMake',}, axis=1)
full_corr.head()
# corr 상관관계 분석
full_corr_df = full_corr.corr()
full_corr_df
# 히트맵
plt.figure(figsize=(10,8))

sns.heatmap(data=full_corr_df,
            annot=True,
            fmt='.2f',
            linewidth=.3,
            cmap='GnBu')
# VIF(분산팽창요인) 확인
# VIF>10 일 경우, 다중공선성이 있다고 판단

import statsmodels as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(full_corr.values, i) for i in range(full_corr.shape[1])]
vif["features"] = full_corr.columns
vif = vif.sort_values("VIF Factor").reset_index(drop=True)
vif
fig = plt.figure(figsize=(4,4))
sns.set_style("whitegrid")

ax = sns.barplot(data=vif, x='features', y='VIF Factor', hue='features', palette='GnBu')

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 0.3,
            round(height, 3), ha = 'center', size = 9)

plt.yticks(np.arange(0, 45, 5))
plt.show()
# 종속변수와 독립변수의 선형 분석
# 산점도 & 선형성 확인
# 이산형 'year'
# 연속형 'price','mileage','tax','engineSize'
df = full2_data.copy(deep=True)
num_colums = df._get_numeric_data().columns.tolist()

num_columns_df = pd.DataFrame(df.loc[:, num_colums])
num_columns_df.head()
figure = plt.figure(figsize=(15,15))

for i in range(1, 7): 
    
    if i <= len(num_columns_df):
        num = num_columns_df.columns[i-1]
        
        if num!='price':
            
            sns.lmplot(data=num_columns_df, x=num, y='price',
                    scatter_kws={"s": 50, "alpha": 1, "color":"skyblue"},
                    line_kws={'color':'khaki'})
            
            plt.ylim(0)
            
            plt.xlabel(f'{num}')
            plt.ylabel('price')
    
    plt.show()
# 변수 변환 별 분포도
# 성능 지표 출력에 필요한 라이브러리리 import
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
# 라벨 인코딩
from sklearn.preprocessing import LabelEncoder
encoding_df = full2_data.copy(deep=True)

LE = LabelEncoder()
encoding_df['model'] = LE.fit_transform(encoding_df['model'])

# 원 핫 인코딩
encoding_df = pd.get_dummies(encoding_df, dtype=int)
print(len(encoding_df.columns))
encoding_df
# 베이스 데이터셋
encoding_num_df = encoding_df.loc[:, ['year', 'price','mileage', 'tax', 'mpg', 'engineSize']]

encoding_num_df.head()
from scipy import stats
from scipy.stats import norm, skew

figure = plt.figure(figsize=(20,15))
rows, cols = 2, 3

for i in range(1, 7): 
    ax = figure.add_subplot(rows, cols, i)
    
    if i <= len(encoding_num_df):
        full2_column = encoding_num_df.columns[i-1]
        x1 = encoding_num_df[full2_column]
        
        sns.distplot(x1, fit=norm)
        (x1_mu, x1_sigma) = norm.fit(x1)
                
        ax.legend([f'Original dist.\n(x1 $\mu=$ {x1_mu:.2f}\nx1 $\sigma=$ {x1_sigma:.2f} )'], loc='best')
        
        plt.xlabel(f'{full2_column}')
        plt.ylabel('Frequency')

    else:
        ax.axis('off')

plt.tight_layout()
plt.show()
encoding_np_df = encoding_df.loc[:, ['year', 'mileage', 'tax', 'mpg', 'engineSize']]

encoding_np_df.head()
from scipy import stats
from scipy.stats import norm, skew

figure = plt.figure(figsize=(20,15))
rows, cols = 2, 3

for i in range(1, 6): 
    ax = figure.add_subplot(rows, cols, i)
    
    if i <= len(encoding_np_df):
        full2_column = encoding_np_df.columns[i-1]
        x1 = encoding_np_df[full2_column]
        
        sns.distplot(x1, fit=norm)
        (x1_mu, x1_sigma) = norm.fit(x1)
                
        ax.legend([f'Original dist.\n(x1 $\mu=$ {x1_mu:.2f}\nx1 $\sigma=$ {x1_sigma:.2f} )'], loc='best')
        
        plt.xlabel(f'{full2_column}')
        plt.ylabel('Frequency')

    else:
        ax.axis('off')

plt.tight_layout()
plt.show()
# 학습 데이터 분포도
x_base = encoding_df.drop(columns='price')
y_base = encoding_df['price']

X_train, X_test, y_train, y_test = train_test_split(x_base, y_base, test_size=0.3, random_state=42)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
X_num_df = X_train.loc[:, ['year','mileage', 'tax', 'mpg', 'engineSize']]

X_num_df.head()
y_num_df = pd.DataFrame(y_train)
y_num_df.head()
from scipy import stats
from scipy.stats import norm, skew

figure = plt.figure(figsize=(20,15))
rows, cols = 2, 3

for i in range(1, 6): 
    ax = figure.add_subplot(rows, cols, i)
    
    if i <= len(X_num_df):
        full2_column = X_num_df.columns[i-1]
        x1 = X_num_df[full2_column]
        
        sns.distplot(x1, fit=norm)
        (x1_mu, x1_sigma) = norm.fit(x1)
                
        ax.legend([f'Trained dist.\n(x1 $\mu=$ {x1_mu:.2f}\nx1 $\sigma=$ {x1_sigma:.2f} )'], loc='best')
        
        plt.xlabel(f'{full2_column}')
        plt.ylabel('Frequency')

    else:
        ax.axis('off')
    

plt.tight_layout()
plt.show()
# 인코딩 전후 비교
full2_data_copy = full2_data.copy(deep=True)
full2_data_copy.reset_index(inplace=True)
full2_data_copy.head()
encoding_df_copy = encoding_df.copy(deep=True)
encoding_df_copy.reset_index(inplace=True)
encoding_df_copy.head()
full2_encoding = pd.merge(full2_data_copy, encoding_df_copy, how='inner', on=full2_data_copy['index'])
full2_encoding.head()
full2_encoding.filter(regex='model').head()
full2_encoding.filter(regex='transmission').head()
full2_encoding.filter(regex='fuelType').head()
full2_encoding.filter(regex='carMake').head()
# 로그 변환 후 변수 분포도
# 라벨 인코딩
log_df = full2_data.copy(deep=True)

LE = LabelEncoder()
log_df['model'] = LE.fit_transform(log_df['model'])

# 원 핫 인코딩
log_df = pd.get_dummies(log_df, dtype=int)
print(len(log_df.columns))
log_df
log_df.info()
log_df[['year', 'price', 'mileage', 'tax', 'mpg', 'engineSize']] = np.log1p(log_df[['year', 'price', 'mileage', 'tax', 'mpg', 'engineSize']])
log_df
log_num_df = log_df.loc[:, ['year', 'price','mileage', 'tax', 'mpg', 'engineSize']]

log_num_df.head()
from scipy import stats
from scipy.stats import norm, skew

figure = plt.figure(figsize=(20,15))
rows, cols = 2, 3
        
for i in range(1, 7): 
    ax = figure.add_subplot(rows, cols, i)
    
    if i <= len(log_num_df):
        log_column = log_num_df.columns[i-1]
        x2 = log_num_df[log_column]
            
        sns.distplot(x2, fit=norm)
        (x2_mu, x2_sigma) = norm.fit(x2)
        
        ax.legend([f'Log dist.\n(x2 $\mu=$ {x2_mu:.2f}\nx2 $\sigma=$ {x2_sigma:.2f} )'], loc='best')
        
        plt.xlabel(f'{log_column}')
        plt.ylabel('Frequency')

    else:
        ax.axis('off')

plt.tight_layout()
plt.show()
# 표준화 후 변수 분포도
# 데이터셋 준비
stand_df = encoding_df.copy(deep=True)

X_stand = stand_df.drop(columns='price')
y_stand = stand_df['price']

xtr_stand, xt_stand, ytr_stand, yt_stand = train_test_split(X_stand, y_stand, test_size=0.3, random_state=42)
print(xtr_stand.shape, xt_stand.shape)
print(ytr_stand.shape, yt_stand.shape)
from sklearn.preprocessing import StandardScaler
 
scaler = StandardScaler()

num_columns = ['year', 'mileage', 'tax', 'mpg', 'engineSize']

scaler.fit(xtr_stand[num_columns])
xtr_stand[num_columns] = scaler.transform(xtr_stand[num_columns])

xt_stand[num_columns] = scaler.transform(xt_stand[num_columns])
stand_num_df = xtr_stand.loc[:, ['year', 'mileage', 'tax', 'mpg', 'engineSize']]
figure = plt.figure(figsize=(20,15))
rows, cols = 2, 3

for i in range(1, 6): 
    ax = figure.add_subplot(rows, cols, i)
    
    if i <= len(stand_num_df):
        stand_column = stand_num_df.columns[i-1]
        x1 = stand_num_df[stand_column]
        
        sns.distplot(x1, fit=norm)
        (x1_mu, x1_sigma) = norm.fit(x1)
                
        ax.legend([f'Stand Scaled dist.\n(x1 $\mu=$ {x1_mu:.2f}\nx1 $\sigma=$ {x1_sigma:.2f} )'], loc='best')
        
        plt.xlabel(f'{stand_column}')
        plt.ylabel('Frequency')

    else:
        ax.axis('off')

plt.tight_layout()
plt.show()
# 정규화 후 변수 분포도
# 데이터셋 준비
minmax_df = encoding_df.copy(deep=True)

X_minmax = minmax_df.drop(columns='price')
y_minmax = minmax_df['price']

xtr_minmax, xt_minmax, ytr_minmax, yt_minmax = train_test_split(X_minmax, y_minmax, test_size=0.3, random_state=42)
print(xtr_minmax.shape, xt_minmax.shape)
print(ytr_minmax.shape, yt_minmax.shape)
from sklearn.preprocessing import MinMaxScaler

scaler2 = MinMaxScaler()

num_columns = ['year', 'mileage', 'tax', 'mpg', 'engineSize']

scaler2.fit(xtr_minmax[num_columns])
xtr_minmax[num_columns] = scaler2.transform(xtr_minmax[num_columns])

xt_minmax[num_columns] = scaler2.transform(xt_minmax[num_columns])
minmax_num_df = xtr_minmax.loc[:, ['year', 'mileage', 'tax', 'mpg', 'engineSize']]
from scipy import stats
from scipy.stats import norm, skew

figure = plt.figure(figsize=(20,15))
rows, cols = 2, 3

for i in range(1, 6): 
    ax = figure.add_subplot(rows, cols, i)
    
    if i <= len(minmax_num_df):
        minmax_column = minmax_num_df.columns[i-1]
        x1 = minmax_num_df[minmax_column]
        
        sns.distplot(x1, fit=norm)
        (x1_mu, x1_sigma) = norm.fit(x1)
                
        ax.legend([f'MinMax Scaled dist.\n(x1 $\mu=$ {x1_mu:.2f}\nx1 $\sigma=$ {x1_sigma:.2f} )'], loc='best')
        
        plt.xlabel(f'{minmax_column}')
        plt.ylabel('Frequency')

    else:
        ax.axis('off')

plt.tight_layout()
plt.show()
### 모델별 결과 지표 시각화
#### 베이스 모델 결과 지표 시각화
base_visual = {'model': ['Linear', 'RandomF', 'XGB', 'LGBM'],
               'MSE': [24426836.294, 5132988.774, 3978017.750, 5825439.919],
               'RMSE': [4942.351, 2265.610, 1994.497, 2413.595],
               'MAE': [2988.287, 1182.199, 1222.250, 1447.402],
               'MAPE': [23.236, 7.397, 7.7, 9.197],
               'R2': [ 0.757, 0.949, 0.954, 0.942]}
base_visual
pd.set_option('display.float_format', '{:.3f}'.format)

base_visual_df = pd.DataFrame(base_visual)

base_visual_df
fig = plt.figure(figsize=(4,4))
sns.set_style("whitegrid")

ax = sns.barplot(data=base_visual_df, x='model', y='MSE', hue='model', palette='GnBu')

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 0.03*1e7,
            height, ha = 'center', size = 9)

plt.yticks(np.arange(0, 3*1e7, 0.25*1e7))
plt.show()
fig = plt.figure(figsize=(4,4))
sns.set_style("whitegrid")

ax = sns.barplot(data=base_visual_df, x='model', y='RMSE', hue='model', palette='GnBu')

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 40,
            height, ha = 'center', size = 9)

plt.yticks(np.arange(0, 5500, 500))
plt.show()
fig = plt.figure(figsize=(4,4))
sns.set_style("whitegrid")

ax = sns.barplot(data=base_visual_df, x='model', y='MAE', hue='model', palette='GnBu')

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 40,
            height, ha = 'center', size = 9)

plt.yticks(np.arange(0, 3300, 300))
plt.show()
fig = plt.figure(figsize=(4,4))
sns.set_style("whitegrid")

ax = sns.barplot(data=base_visual_df, x='model', y='MAPE', hue='model', palette='GnBu')

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 0.2,
            height, ha = 'center', size = 9)

plt.yticks(np.arange(0, 24, 2))
plt.show()
fig = plt.figure(figsize=(4,4))
sns.set_style("whitegrid")

ax = sns.barplot(data=base_visual_df, x='model', y='R2', hue='model', palette='GnBu')

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 0.005,
            height, ha = 'center', size = 9)

plt.yticks(np.arange(0, 1, 0.1))
plt.show()
#### 데이터 스케일링 결과 지표 시각화
##### mape 결과
scale_mape = {}
scale_mape['Linear'] = {'Base':22.236, 'Log': 14.008, 'Standard':22.236, 'MinMax': 22.236}
scale_mape['RandomF'] = {'Base':7.397, 'Log': 7.270, 'Standard':7.355, 'MinMax': 7.399}
scale_mape['XGB'] = {'Base':7.700, 'Log': 7.355, 'Standard':7.756, 'MinMax': 7.756}
scale_mape['LGBM'] = {'Base':9.197, 'Log': 8.642, 'Standard':9.209, 'MinMax': 9.173}
pd.set_option('display.float_format', '{:.3f}'.format)

scale_mape_df = pd.DataFrame(scale_mape)

scale_mape_df
scale_mape_df = scale_mape_df.unstack(level=1)
scale_mape_df_i = scale_mape_df.reset_index()
scale_mape_df_i
scale_mape_df_i.columns = ['model', 'scaled', 'value']
scale_mape_df_i.columns
fig = plt.figure(figsize=(8,8))
sns.set_style("whitegrid")

ax = sns.barplot(data=scale_mape_df_i, x='model', y='value', hue='scaled', palette='GnBu')

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 0.2,
            height, ha = 'center', size = 7)
    
plt.yticks(np.arange(0, 24, 2))

plt.show()
##### R2 결과
scale_r2 = {}
scale_r2['Linear'] = {'Base':0.757, 'Log': 0.846, 'Standard':0.777, 'MinMax': 0.777}
scale_r2['RandomF'] = {'Base':0.949, 'Log': 0.950, 'Standard':0.949, 'MinMax': 0.949}
scale_r2['XGB'] = {'Base':0.954, 'Log': 0.958, 'Standard':0.954, 'MinMax': 0.954}
scale_r2['LGBM'] = {'Base':0.942, 'Log': 0.939, 'Standard':0.942, 'MinMax': 0.942}
pd.set_option('display.float_format', '{:.3f}'.format)

scale_r2_df = pd.DataFrame(scale_r2)

scale_r2_df
scale_r2_df = scale_r2_df.unstack(level=1)
scale_r2_df_i = scale_r2_df.reset_index()
scale_r2_df_i
scale_r2_df_i.columns = ['model', 'scaled', 'value']
scale_r2_df_i.columns
fig = plt.figure(figsize=(8,8))
sns.set_style("whitegrid")

ax = sns.barplot(data=scale_r2_df_i, x='model', y='value', hue='scaled', palette='GnBu')

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 0.01,
            height, ha = 'center', size = 7)
    
plt.yticks(np.arange(0, 1.02, 0.06))
plt.legend(loc="lower right")

plt.show()
#### 추가 스케일링 결과 지표 시각화
##### mape 결과
add_scale_mape = {}
add_scale_mape['Linear'] = {'Base':23.236, 'Log': 14.001, 'Log_Standard':14.001, 'Log_MinMax': 14.001, 'Log_Robust': 14.001}
add_scale_mape['RandomF'] = {'Base':7.397, 'Log': 7.270, 'Log_Standard':7.270, 'Log_MinMax': 7.272, 'Log_Robust': 7.271}
add_scale_mape['XGB'] = {'Base':7.700, 'Log': 7.419, 'Log_Standard':8.700, 'Log_MinMax': 7.419, 'Log_Robust': 7.419}
add_scale_mape['LGBM'] = {'Base':9.197, 'Log': 8.700, 'Log_Standard':8.700, 'Log_MinMax': 8.698, 'Log_Robust': 8.695}
pd.set_option('display.float_format', '{:.3f}'.format)

add_scale_mape_df = pd.DataFrame(add_scale_mape)

add_scale_mape_df
add_scale_mape_df = add_scale_mape_df.unstack(level=1)
add_scale_mape_df_i = add_scale_mape_df.reset_index()
add_scale_mape_df_i
add_scale_mape_df_i.columns = ['model', 'scaled', 'value']
add_scale_mape_df_i.columns
fig = plt.figure(figsize=(8,8))
sns.set_style("whitegrid")

ax = sns.barplot(data=add_scale_mape_df_i, x='model', y='value', hue='scaled', palette='GnBu')

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 0.2,
            height, ha = 'center', size = 7)

plt.yticks(np.arange(0, 24, 2))
plt.show()
##### R2 결과
add_scale_r2 = {}
add_scale_r2['Linear'] = {'Base':0.757, 'Log': 0.842, 'Log_Standard':0.842, 'Log_MinMax': 0.842, 'Log_Robust': 0.842}
add_scale_r2['RandomF'] = {'Base':0.949, 'Log': 0.950, 'Log_Standard':0.950, 'Log_MinMax': 0.950, 'Log_Robust': 0.950}
add_scale_r2['XGB'] = {'Base':0.954, 'Log': 0.950, 'Log_Standard':0.927, 'Log_MinMax':0.950, 'Log_Robust': 0.950}
add_scale_r2['LGBM'] = {'Base':0.942, 'Log': 0.927, 'Log_Standard':0.927, 'Log_MinMax': 0.929, 'Log_Robust': 0.928}
pd.set_option('display.float_format', '{:.3f}'.format)

add_scale_r2_df = pd.DataFrame(add_scale_r2)

add_scale_r2_df
add_scale_r2_df = add_scale_r2_df.unstack(level=1)
add_scale_r2_df_i = add_scale_r2_df.reset_index()
add_scale_r2_df_i
add_scale_r2_df_i.columns = ['model', 'scaled', 'value']
add_scale_r2_df_i.columns
fig = plt.figure(figsize=(8,8))
sns.set_style("whitegrid")

ax = sns.barplot(data=add_scale_r2_df_i, x='model', y='value', hue='scaled', palette='GnBu')

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 0.01,
            height, ha = 'center', size = 7)
    
plt.yticks(np.arange(0, 1.02, 0.06))
plt.legend(loc="lower right")

plt.show()
#### 이상치 전처리 후 모델 결과 지표 시각화
# year = 1975 제외
year_1975 = {'model': ['Base_Log_Robust', 'Log', 'Log_Standard','Log_MinMax', 'Log_Robust'],
               'MAPE': [ 7.271, 7.270, 7.270, 7.272, 7.272],
               'R2': [0.950, 0.950, 0.950, 0.950, 0.950]}
year_1975
pd.set_option('display.float_format', '{:.3f}'.format)

year_1975_df = pd.DataFrame(year_1975)

year_1975_df
fig = plt.figure(figsize=(6, 4))
fig.set_facecolor('white') 
ax = fig.add_subplot()  #서브플롯 생성

ax.bar(year_1975_df['model'], year_1975_df['MAPE'], label='MAPE',color='skyblue', alpha=0.7)
ax.set_ylabel('MAPE')
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 0.05,
            height, ha = 'center', size = 5)

ax2 = ax.twinx()
ax2.plot(year_1975_df['model'], year_1975_df['R2'], label='R2',color='orange', marker='o', linewidth=1)
ax2.set_ylabel('R2')

ax.legend(loc = 'upper left')
ax2.legend(loc = 'upper right')

plt.title('Year=1975 Delete')
plt.grid(False)
plt.show()
# engineSize=0 제외
engine_0 = {'model': ['Base_Log_Robust', 'Log', 'Log_Standard','Log_MinMax', 'Log_Robust'],
               'MAPE': [  7.271, 7.397, 7.183, 7.183, 7.183],
               'R2': [ 0.950, 0.949, 0.963, 0.963, 0.963]}

pd.set_option('display.float_format', '{:.3f}'.format)

engine_0_df = pd.DataFrame(engine_0)

engine_0_df
fig = plt.figure(figsize=(6, 4))
fig.set_facecolor('white') 
ax = fig.add_subplot()  #서브플롯 생성

ax.bar(engine_0_df['model'], engine_0_df['MAPE'], label='MAPE',color='olivedrab', alpha=0.7)
ax.set_ylabel('MAPE')
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2 - 0.16,
            height + 0.05,
            height, ha = 'center', size = 5)

ax2 = ax.twinx()
ax2.plot(engine_0_df['model'], engine_0_df['R2'], label='R2',color='firebrick', marker='o', linewidth=1)
ax2.set_ylabel('R2')

ax.legend(loc = 'upper left')
ax2.legend(loc = 'lower right')

plt.title('EngineSize=0 Delete')
plt.grid(False)
plt.show()
#### 정규화 제외 후 모델 결과 지표 시각화
# year = 1975 제외
year_no = {'model': ['Base_Log_Robust', 'Log', 'Log_Standard','Log_MinMax', 'Log_Robust'],
               'MAPE': [ 7.271, 7.270, 7.270, 7.271, 7.270],
               'R2': [ 0.950, 0.950, 0.950, 0.950, 0.950]}
year_no
pd.set_option('display.float_format', '{:.3f}'.format)

year_no_df = pd.DataFrame(year_no)

year_no_df
fig = plt.figure(figsize=(6, 4))
fig.set_facecolor('white') 
ax = fig.add_subplot()  #서브플롯 생성

ax.bar(year_no_df['model'], year_no_df['MAPE'], label='MAPE',color='skyblue', alpha=0.7)
ax.set_ylabel('MAPE')
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 0.05,
            height, ha = 'center', size = 5)

ax2 = ax.twinx()
ax2.plot(year_no_df['model'], year_no_df['R2'], label='R2',color='orange', marker='o', linewidth=1)
ax2.set_ylabel('R2')

ax.legend(loc = 'upper left')
ax2.legend(loc = 'upper right')

plt.title('Year Feature Exception')
plt.grid(False)
plt.show()
# engineSize 정규화화 제외
engine_no = {'model': ['Base_Log_Robust', 'Log', 'Log_Standard','Log_MinMax', 'Log_Robust'],
               'MAPE': [  7.271, 7.270, 7.270, 7.272, 7.272],
               'R2': [0.950, 0.950, 0.950, 0.950, 0.950]}

pd.set_option('display.float_format', '{:.3f}'.format)

engine_no_df = pd.DataFrame(engine_no)

engine_no_df
fig = plt.figure(figsize=(6, 4))
fig.set_facecolor('white') 
ax = fig.add_subplot()  #서브플롯 생성

ax.bar(engine_no_df['model'], engine_no_df['MAPE'], label='MAPE',color='olivedrab', alpha=0.7)
ax.set_ylabel('MAPE')
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2,
            height + 0.05,
            height, ha = 'center', size = 5)

ax2 = ax.twinx()
ax2.plot(engine_no_df['model'], engine_no_df['R2'], label='R2',color='firebrick', marker='o', linewidth=1)
ax2.set_ylabel('R2')

ax.legend(loc = 'upper left')
ax2.legend(loc = 'upper right')

plt.title('EngineSize Feature Exception')
plt.grid(False)
plt.show()

#### 혼합 전처리 모델 결과 지표 시각화
# year = 1975 제외
year_engine = {'model': ['Base_Log_Robust', 'Log', 'Log_Standard','Log_MinMax', 'Log_Robust'],
               'MAPE': [ 7.271,  7.397, 7.160, 7.160, 7.161],
               'R2': [ 0.950, 0.949, 0.959, 0.959, 0.959]}
year_engine
pd.set_option('display.float_format', '{:.3f}'.format)

year_engine_df = pd.DataFrame(year_engine)

year_engine_df
fig = plt.figure(figsize=(6, 4))
fig.set_facecolor('white') 
ax = fig.add_subplot()  #서브플롯 생성

ax.bar(year_engine_df['model'], year_engine_df['MAPE'], label='MAPE',color='skyblue', alpha=0.7)
ax.set_ylabel('MAPE')
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2. - 0.16,
            height + 0.05,
            height, ha = 'center', size = 5)

ax2 = ax.twinx()
ax2.plot(year_engine_df['model'], year_engine_df['R2'], label='R2',color='firebrick', marker='o', linewidth=1)
ax2.set_ylabel('R2')

ax.legend(loc = 'upper left')
ax2.legend(loc = 'lower right')

plt.title('EngineSize=0 Delete & Year Feature Exception')
plt.grid(False)
plt.show()
#### 최종 모델 결과 지표 시각화
# year = 1975 제외
final = {'model': ['Base', 'Year_Engine', 'HyperParameter'],
         'MSE': [ 5132988.774, 3533081.493, 3961250.170],
         'RMSE': [2265.610, 1879.649, 1990.289],
         'MAE': [1182.199, 1146.450, 1130.304],
         'MAPE': [7.397, 7.181, 6.997],
         'R2': [ 0.949, 0.963, 0.959],
         'EVL': [0.994, 0.994, 0.986]}
final
pd.set_option('display.float_format', '{:.3f}'.format)

final_df = pd.DataFrame(final)

final_df
fig = plt.figure(figsize=(4, 4))
fig.set_facecolor('white') 
ax = fig.add_subplot()  #서브플롯 생성

ax.bar(final_df['model'], final_df['MAPE'], label='MAPE',color='skyblue', alpha=0.7)
ax.set_ylabel('MAPE')
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2. - 0.16,
            height + 0.05,
            height, ha = 'center', size = 5)

ax2 = ax.twinx()
ax2.plot(final_df['model'], final_df['R2'], label='R2',color='firebrick', marker='o', linewidth=1)
ax2.set_ylabel('R2 & EVL')

ax.legend(loc = 'center left')
ax2.legend(loc = 'lower right')

plt.legend()
plt.title('Final Random Forest Models')
plt.grid(False)
plt.show()
feature_imp = {'feature': ['tm_manual', 'year', 'engineSize', 'model', 'mileage'],
               'impact': [0.334, 0.323, 0.172, 0.059, 0.039]}

pd.set_option('display.float_format', '{:.3f}'.format)

feature_imp_df = pd.DataFrame(feature_imp)

feature_imp_df
fig = plt.figure(figsize=(4, 4))
fig.set_facecolor('white') 

plt.bar(feature_imp_df['feature'], feature_imp_df['impact'], label='Importance',color='skyblue', alpha=0.7)
for p in .patches:
    height = p.get_height()
    plt.text(p.get_x() + p.get_width() / 2.,
            height,
            height, ha = 'center', size = 0.5)

plt.title('Variable Importance')
plt.grid(True)
plt.show()
fig = plt.figure(figsize=(6,6))
sns.set_style("whitegrid")

ax = sns.barplot(data=feature_imp_df, x='feature', y='impact', hue='feature', palette='GnBu')

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 0.002,
            height, ha = 'center', size = 7)
    
plt.yticks(np.arange(0, 0.375, 0.025))
plt.show()