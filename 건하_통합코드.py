"""
<Python 문서 전체 설명>
1. 전체 순서
    1. 필요 라이브러리, 데이터 불러오기
    2. EDA 및 데이터 시각화
    3. 이상치 처리
    4. 데이터 인코딩
    5. 데이터 분할
    6. 데이터 스케일링
    7. 모델 학습 및 평가
        7-1. 선형회귀
        7-2. 랜덤 포레스트
        7-3. XGBoost
        7-4. LightGBM
    8. 결과 비교
    9. 결과 해석
    10. 모델 저장 및 로딩
2. #: 코드 설명, ##: 대제목, ###: 소제목
    ##과 ###은 마크다운 셀에서 실행, 일반 셀과 구분하기 위해 ***으로 감싸줌
3. 수정하기 쉽도록 문단 간의 띄어쓰기는 최종 파이썬 파일을 정리할 때 할 예정
    파이썬 파일은 코드를 전체적으로 볼 때만 사용
"""
## ***7-1. 선형회귀***
## ***7-2. 랜덤 포레스트***
## ***7-3. XGBoost***
## ***7-4. LightGBM***
## ***8. 결과 비교***
## ***9. 결과 해석***
## ***10. 모델 저장 및 로딩***
## ***1. 필요 라이브러리 및 데이터 불러오기***
# warning 제거
import warnings
warnings.filterwarnings('ignore')
# 필요 라이브러리 가져오기
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
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
# full_data 내보내기
# full_data.to_csv('full_data.csv', index=False) # 파일 실행마다 파일 생성을 방지하기 위해 주석 처리
## ***2. EDA 및 데이터 시각화***
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
# 다중공선성(독립변수 간의 상관관계) 확인
full2_data_numeric = full2_data[['year', 'mileage', 'tax', 'mpg', 'engineSize']]
sns.heatmap(data=full2_data_numeric.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.show()
## ***3. 이상치 처리***
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
# year 이상치, 2060년 데이터 1개 삭제
print(full2_data.shape)
cond1 = (full2_data['year'] == 2060)
full2_data = full2_data[~cond1]
print(full2_data.shape)
## ***4. 데이터 인코딩***
# 인코딩 전 컬럼 확인
print(full2_data.select_dtypes(exclude=[int, float]).columns) # 수치형 자료가 아닌 열 4개
full2_data.head(3)
# 명목형 변수 별 고유값 개수 확인: 값 많으면 라벨, 적으면 원 핫 인코딩(모델의 학습 속도 고려)
print(full2_data[['model', 'transmission', 'fuelType', 'carMake']].nunique())
# 명목형 변수의 고유값 별 개수 확인1
vc = full2_data['model'].value_counts()

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
LE = LabelEncoder()
encoding_df=full2_data.copy()
encoding_df['model'] = LE.fit_transform(encoding_df['model'])
encoding_df['model'].head(3)
# 데이터 분할
from sklearn.model_selection import train_test_split
X = encoding_df.drop(columns='price')
y = encoding_df[['price']]
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.30, random_state=42)
# model 외 3개 명목형 변수 원 핫 인코딩
from sklearn.preprocessing import OneHotEncoder

str_list = ['transmission', 'fuelType', 'carMake']

for i in str_list:
  ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
  # fit_transform은 train에만 사용하고 test에는 학습된 인코더에 fit만 해야한다
  train_transmission = ohe.fit_transform(X_train[[i]]) # numpy.array
  train_transmission_df = pd.DataFrame(train_transmission, columns=[i + col for col in ohe.categories_[0]])
  X_train = pd.concat([X_train.reset_index(drop=True), train_transmission_df], axis=1)
  del X_train[i]
  # 학습된 인코더에 test를 fit 합니다.
  X_test_transmission = ohe.transform(X_test[[i]])
  X_test_transmission_df = pd.DataFrame(X_test_transmission, columns=[i + col for col in ohe.categories_[0]])
  X_test = pd.concat([X_test.reset_index(drop=True), X_test_transmission_df], axis=1)
  del X_test[i]

X_train

## ***5. 데이터 분할***
## ***6. 데이터 스케일링***
## ***7. 모델 학습 및 평가***
# 전처리 된, 파일 불러오기
full_2_data = pd.read_csv('full_2_data.csv')
full_2_data.head()
# 명목변수 = ['model', 'transmission', 'fuelType', 'carMake']
# 라벨 인코딩 (추후에 빈도 인코딩으로 변경 가능)
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
full_2_data['model'] = LE.fit_transform(full_2_data['model'])
# 원 핫 인코딩
encoding_df = pd.get_dummies(full_2_data)
print(len(encoding_df.columns))
encoding_df
# 데이터셋 준비(X,y)
X = encoding_df.drop(columns='price')
y = encoding_df['price']
# 데이터셋 분할
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
# 랜덤 포레스트 회귀 모델 학습
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
# 랜덤 포레스트 R2-score
print(rfr.score(X_train, y_train))
print(rfr.score(X_test, y_test))
# MAPE
y_pred_train = rfr.predict(X_train)
y_pred_test = rfr.predict(X_test)

from sklearn.metrics import mean_absolute_percentage_error
MAPE_train = mean_absolute_percentage_error(y_train, y_pred_train) * 100
MAPE_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100
print(MAPE_train)
print(MAPE_test)
