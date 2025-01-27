# ▶ 경로 설정 (※강의자료가 위치에 있는 경로 확인)
import os
os.chdir('/content/drive/MyDrive/스파르타_COLAB/250124-250206 심화 프로젝트/used_car copy_original')
os.getcwd()

# 비교 컬럼 값 확인
full_data['model'].unique()

# cclass 비교
# cclass.csv 파일 불러오기 및 정보 확인
cclass_df = pd.read_csv('cclass.csv')
print(cclass_df.info())
print(cclass_df['model'].unique())
print(cclass_df.isnull().sum())

# full_data의 cclass 컬럼 값 확인
cond1 = full_data['model']==' C Class'
cclass_full = full_data.loc[cond1]
print(cclass_full.info())
print(cclass_full.isnull().sum())
print(cclass_full['carMake'].unique())

# full_data의 carMake = cclass 정보 확인
cond2 = cclass_full['carMake'] == 'cclass'
cclass_make = cclass_full.loc[cond2]
print(cclass_make.info())

# focus 비교
# focus.csv 파일 불러오기 및 정보 확인
focus_df = pd.read_csv('focus.csv')
print(focus_df.info())
print(focus_df['model'].unique())
print(focus_df.isnull().sum())

# full_data의 focus 컬럼 값 확인
cond4 = focus_full['carMake']=='focus'
focus_make = focus_full.loc[cond4]
print(focus_full.info())
print(focus_full.isnull().sum())
print(focus_full['carMake'].unique())

# full_data의 carMake = focus 정보 확인
cond4 = focus_full['carMake']=='focus'
focus_make = focus_full.loc[cond4]
print(focus_make.info())

# full2_data 와 비교하기
# full2_data의 model = cclass 정보 확인
cond1_1 = full2_data['model'] == ' C Class'
cclass_full2 = full2_data.loc[cond1_1]
print(cclass_full2.info())

# cclass.csv 파일 정보와 중복값 개수 조회
cclass_df_full = pd.merge(cclass_full2, cclass_df, on = ['model', 'year', 'price', 'transmission', 'mileage', 'fuelType', 'engineSize'], how = "inner")
print(cclass_df_full.info())

# full2_data의 model = focus 정보 확인
cond3_1 = full2_data['model']==' Focus'
focus_full2 = full2_data.loc[cond3_1]
print(focus_full2.info())

# focus.csv 파일 정보와 중복값 개수 조회
focus_df_full = pd.merge(focus_full2, focus_df, on = ['model', 'year', 'price', 'transmission', 'mileage', 'fuelType', 'engineSize'], how = "inner")
print(focus_df_full.info())
