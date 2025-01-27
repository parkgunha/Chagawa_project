# 전체 순서: 1. 데이터 읽기 / 2. 데이터셋 준비하기 / 3. 데이터셋 분할하기

# 1. 데이터 읽기

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
input_path = os.getcwd() + '/used_car copy/'  # 데이터셋이 저장된 경로
file_paths = []

for dirname, _, filenames in os.walk(input_path):
    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        if filename.endswith('.csv'):  # CSV 파일만 선택
            file_paths.append(file_path)

print(f"총 {len(file_paths)}개의 CSV 파일을 발견했습니다.")

# CSV 파일 읽기 및 병합
data_frames = []

for file_path in file_paths:
    df = pd.read_csv(file_path)
    print(f"{file_path} 파일 읽기 완료. 데이터 크기: {df.shape}")
    df["carMake"] = file_path.split('/')[-1].split('.')[0]
    data_frames.append(df)

# 모든 데이터프레임을 하나로 병합
if data_frames:
    full2_data = pd.concat(data_frames, ignore_index=True)
    print(f"병합된 데이터 크기: {full2_data.shape}")
else:
    print("CSV 파일이 없습니다.")

# 데이터프레임 확인
print("데이터의 첫 5행 미리보기:")
print(full2_data.head())

# full2_data 내보내기
full2_data.to_csv('full2_data.csv', index=False)

# full2_data로 선택: cclass, focus 데이터가 full2_data에 전부 포함되어있음

# full2_data.csv를 df로 변환
full2_data =  pd.read_csv('full2_data') # git hub 같은 디렉토리에 저장되어있음.

# 2. full2_data 데이터셋 준비하기: 전처리

# <결측값 제거>
# tax열 결측값 제거: tax 와 tax(£) 컬럼 합치기
full2_data['tax'].fillna(full2_data['tax(£)'], inplace=True)
full2_data.drop(columns='tax(£)', inplace=True)
full2_data.isna().sum()

# <명목형 변수 공백 삭제>
str_list = ['model', 'transmission', 'fuelType', 'carMake'] # 명목변수
for i in str_list:
    full2_data[i] = full2_data[i].str.strip()
print(full2_data['model'].unique()[:20])

# <이상치 제거>
# year 범위 설정: 2014 이상 2020년 이하
full2_data = full2_data[(full2_data['year'] >= 2014) &(full2_data['year'] <= 2020)]

# IQR 기준으로 year 이상치 탐지
Q1 = full2_data["year"].quantile(0.25)
Q3 = full2_data["year"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = full2_data[(full2_data["year"] < lower_bound) | (full2_data["year"] > upper_bound)]
print(outliers)


# year boxplot 그려보기
plt.boxplot(full2_data['year'])
plt.show()
full2_data['year'].describe()

# 모델 별로 정상범위와 이상치를 확인하기: boxplot 그리기
unique_model = full2_data['model'].unique()
count_nums=0
for i in unique_model:
    model_year = full2_data[full2_data['model'] == i]['year']
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
full2_data_cnt = full2_data.groupby(by='year').agg('count')
full2_data_cnt

# year의 누적합과, 누적비율을 세기: 2014년으로 결정, 데이터의 95% 확인할 수 있음
full2_data_cnt['cumsum'] = full2_data_cnt['model'].cumsum()
full2_data_cnt['cumsum_ratio'] = full2_data_cnt['cumsum'] / 99187
full2_data_cnt[['model', 'cumsum', 'cumsum_ratio']]
sns.lineplot(data=full2_data_cnt, x='year', y='cumsum_ratio')
plt.show()


# 독립변수가 여러개라서 변수간의 상관성 확인 필요-> 상관분석
num_colums = full2_data._get_numeric_data().columns.tolist()
correlation_matrix = full2_data[num_colums].corr()
sns.pairplot(full2_data[num_colums])
plt.show()
sns.heatmap(correlation_matrix, annot=True)
plt.show()


## 아래에는 현수님께 여쭤보고 수정 예정

#명목형 인코딩
#str_list = ['model', 'transmission', 'fuelType', 'carMake'] # 명목변수

# 라벨 인코딩 => 빈도 인코딩
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
incoding_df=df.copy()
incoding_df['model'] = LE.fit_transform(incoding_df['model'])

# 원 핫 인코딩
# 명목형 'model', 'transmission', 'fuelType', 'carMake'
incoding_df = pd.get_dummies(incoding_df, columns = ['transmission', 'fuelType', 'carMake'])
incoding_df.head()

