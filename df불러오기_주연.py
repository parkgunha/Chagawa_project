# ▶ Warnings 제거
import warnings
warnings.filterwarnings('ignore')


# ▶ 경로 설정 (※강의자료가 위치에 있는 경로 확인)
import os
os.chdir("I:\PSAutoRecover\hamju\공모전 및 대외활동\2024 스파르타 코딩클럽\본캠프_CH03_데이터분석 입문\심화 프로젝트\used_car copy\used_car copy")
os.getcwd()

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