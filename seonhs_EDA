#구글 colab 작업 2025--01-27
#Python 3.11.11

#1.파일 업로드
from google.colab import files
files.upload()

#2. 다운로드한 파일 확인 및 압축 해제
import zipfile
import os

#3. 압축 해제
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall("database")

#4. 데이터 파일 확인
input_path = '/content/database/used_car copy'  # 데이터셋이 저장된 경로
file_paths = []
for dirname, _, filenames in os.walk(input_path):
    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        if filename.endswith('.csv'):  # CSV 파일만 선택
            file_paths.append(file_path)
print(f"총 {len(file_paths)}개의 CSV 파일을 발견했습니다.")

#5. CSV 파일 읽기 및 병합
import pandas as pd
data_frames = []

for file_path in file_paths:
    df = pd.read_csv(file_path)
    print(f"{file_path} 파일 읽기 완료. 데이터 크기: {df.shape}")
    df["carMake"] = file_path.split('/')[-1].split('.')[0]
    data_frames.append(df)

#6 모든 데이터프레임을 하나로 병합
if data_frames:
    full_data = pd.concat(data_frames, ignore_index=True)
    print(f"병합된 데이터 크기: {full_data.shape}")
else:
    print("CSV 파일이 없습니다.")

# 데이터프레임 확인
print("데이터의 첫 5행 미리보기:")
print(full_data.head())

# full2_data은 따로 업로드
file_path = '/content/full2_data.csv'
full2_data =  pd.read_csv(file_path)

# 명목형 변수 공백 삭제
df = full2_data.copy()
str_list = ['model', 'transmission', 'fuelType', 'carMake'] # 명목변수
for i in str_list:
  df[i] = df[i].str.strip()
full3_data = df.copy()
full3_data.to_csv("full3_data.csv")

#‘tax’ 와 ‘tax(£)’ 컬럼 합치기
df = full3_data.copy()
if "tax(£)" in df.columns:
  df['tax'] = pd.concat([df['tax'], df['tax(£)']], axis=0).dropna()
  del df['tax(£)']
df.info()
full4_data = df.copy()
full4_data.to_csv("full4_data.csv")

# EDA:year 범위 설정
df = df[(df['year'] >= 2014) &(df['year'] <= 2020)] # 2014이상 2020년 이하

# 독립변수가 여러개라서 변수간의 상관성 확인 필요-> 상관분석
import seaborn as sns
import matplotlib.pyplot as plt
num_colums = df._get_numeric_data().columns.tolist()
correlation_matrix = df[num_colums].corr()
sns.pairplot(df[num_colums])
plt.show()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

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
