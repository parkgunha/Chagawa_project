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
    df["carMake"] = file_path.split('/')[-1].split('.')[0]
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
### ***EDA를 위한 값 수정 및 채우기***
# 전처리를 위한 복사본 만들기
full_data2 = full_data.copy(deep=True)

# 명목형 변수 값의 공백을 삭제
str_list = ['model', 'transmission', 'fuelType', 'carMake'] # 명목변수
for i in str_list:
    full_data2[i] = full_data2[i].str.strip()
print(full_data2['model'].unique()[:20])
# tax열 결측치 채우기: tax 와 tax(£) 컬럼 합치기
full_data2['tax'].fillna(full_data2['tax(£)'], inplace=True)
full_data2.drop(columns='tax(£)', inplace=True)
full_data2.isna().sum()
## ***3. 이상치 처리***
# 모델 별 이상치 확인 
unique_model = full_data['model'].unique()
count_nums=0
for i in unique_model:
    model_year = full_data[full_data['model'] == i]['year']
    if any(x < 2000 for x in model_year):
        count_nums += 1
        year_before_2000 = [x for x in model_year if x < 2000]
        print(f'model name:{i}, count:{len(model_year)}')
        print(f'2000년 이전 year 수:{len(year_before_2000)}')
        plt.boxplot(model_year)
        plt.show()
    elif any(x > 2020 for x in model_year):
        count_nums += 1
        year_after_2020 = [x for x in model_year if x > 2020]
        print(f'model name:{i}, count:{len(model_year)}')
        print(f'2000년 이후 year 수:{len(year_before_2000)}')
        plt.boxplot(model_year)
        plt.show()
print(f'count_nums:{count_nums}')
# year별 개수 세기
full_data_cnt = full_data.groupby(by='year').agg('count')
full_data_cnt
# year의 누적합과, 누적비율을 세기: 2014년~2020년은 데이터의 95%
full_data_cnt['cumsum'] = full_data_cnt['model'].cumsum()
full_data_cnt['cumsum_ratio'] = full_data_cnt['cumsum'] / 99187
full_data_cnt[['model', 'cumsum', 'cumsum_ratio']]
sns.lineplot(data=full_data_cnt, x='year', y='cumsum_ratio')
plt.show()
# year 범위 설정: 2014 이상 2020년 이하
# full_data = full_data[(full_data['year'] >= 2014) &(full_data['year'] <= 2020)]
# IQR 기준으로 year 이상치 탐지
Q1 = full_data["year"].quantile(0.25)
Q3 = full_data["year"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = full_data[(full_data["year"] < lower_bound) | (full_data["year"] > upper_bound)]
print(outliers)
# year boxplot 그려보기
plt.boxplot(full_data['year'])
plt.show()
full_data['year'].describe()
## ***4. 데이터 인코딩***
# 명목변수 = ['model', 'transmission', 'fuelType', 'carMake']
# 라벨 인코딩 (추후에 빈도 인코딩으로 변경 가능)
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
incoding_df = full_data.copy()
incoding_df['model'] = LE.fit_transform(incoding_df['model'])
# 원 핫 인코딩
incoding_df = pd.get_dummies(incoding_df, columns = ['transmission', 'fuelType', 'carMake'])
incoding_df.head()
# 다중공선성(독립변수 간의 상관관계) 확인
full_data_numeric = full_data[['year', 'mileage', 'tax', 'mpg', 'engineSize']]
sns.heatmap(data=full_data_numeric.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.show()

## ***5. 데이터 분할***
## ***6. 데이터 스케일링***
## ***7. 모델 학습 및 평가***
## ***7-1. 선형회귀***
## ***7-2. 랜덤 포레스트***
## ***7-3. XGBoost***
## ***7-4. ?***
## ***8. 결과 비교***
## ***9. 결과 해석***
## ***10. 모델 저장 및 로딩***