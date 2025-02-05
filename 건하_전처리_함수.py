# year 이상치 제거: 출시년도가 데이터를 뽑은 시점인 2020년보다 더 최신인 데이터 
def delete_weird_year(data):
    print(data.shape)
    cond1 = data['year'] > 2025
    data = data[~cond1]
    print(data.shape)
    data.reset_index(drop=True, inplace=True) # 행과 함께 중간의 index 1개가 삭제됨. index를 초기화

# 명목변수 = ['model', 'transmission', 'fuelType', 'carMake']

# 'model' 라벨 인코딩 (추후에 빈도 인코딩으로 변경 가능)
def label_encoding(data):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    model_encoded = le.fit_transform(data['model'])
    data['model_encoded'] = model_encoded
    data.drop(columns='model', inplace=True)
    print(data.shape)

# model 외 3개 명목형 변수 원 핫 인코딩: 18개 인코딩 열 생성, 기존 명목형 열 3개 제외 -> 15개 추가 열 생성
def one_hot_encoding(data):
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd

    str_list = ['transmission', 'fuelType', 'carMake']

    for col in str_list:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        values = ohe.fit_transform(data[[col]])
        columns = [f'{col}_{cat}' for cat in ohe.categories_[0]] # 원래컬럼명 + '_' + 리스트에서 넘파이 배열[0]: 3개 합치기
        df = pd.DataFrame(data=values, columns=columns)
        data = pd.concat([data, df], axis=1)
        data.drop(columns=col, inplace=True)

    return data
