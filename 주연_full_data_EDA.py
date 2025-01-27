# ▶ Warnings 제거
import warnings
warnings.filterwarnings('ignore')

# ▶ Google drive mount or 폴더 클릭 후 구글드라이브 연결
from google.colab import drive
drive.mount('/content/drive')

# 필수 라이브러리 임포트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    full_data = pd.concat(data_frames, ignore_index=True)
    print(f"병합된 데이터 크기: {full_data.shape}")
else:
    print("CSV 파일이 없습니다.")

# 데이터프레임 확인
print("데이터의 첫 5행 미리보기:")
print(full_data.head())

# full_data 저장
full_data.to_csv('full_data.csv', index=False)

# full_data EDA
print(full_data.info())
print(full_data.isnull().sum())
print(full_data.describe(include='all'))

# 01. year_boxplot
plt.figure(figsize=(4, 6))

plt.boxplot(full_data['year'])
plt.yticks(np.arange(1965, 2065, 5))
plt.title('Year Box Plot')

# 02. price_boxplot
plt.figure(figsize=(4, 6))

plt.boxplot(full_data['price'])
plt.yticks(np.arange(400, 170400, 10000))
plt.title('Price Box Plot') 

print(full_data['price'].median())

# 03. mileage_boxplot
plt.figure(figsize=(4, 6))

plt.boxplot(full_data['mileage'])
plt.yticks(np.arange(0, 360000, 30000))
plt.title('Mileage Box Plot')

# 04. tax_boxplot
plt.figure(figsize=(4, 6))

plt.boxplot(full_data['tax'].dropna())
plt.yticks(np.arange(0, 650, 50))
plt.title('Tax Box Plot')

# 04. tax(£)_boxplot
plt.figure(figsize=(4, 6))

plt.boxplot(full_data['tax(£)'].dropna())
plt.yticks(np.arange(0, 650, 50))
plt.title('Tax(£) Box Plot')

# 05. mpg_boxplot
plt.figure(figsize=(4, 6))

plt.boxplot(full_data['mpg'].dropna())
plt.yticks(np.arange(20, 490, 20))
plt.title('Mpg Box Plot')

# 06. engineSize_boxplot
plt.figure(figsize=(4, 6))

plt.boxplot(full_data['engineSize'])
plt.yticks(np.arange(0, 6, 1))
plt.title('EngineSize Box Plot')
