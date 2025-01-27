# year 컬럼 EDA
# year별 price 평균값 비교
full2_year_price = full2_data.groupby('year')['price'].mean()
full2_yp = pd.DataFrame(full2_year_price)
full2_yp.reset_index(drop = False, inplace=True)
full2_yp.head()

fig = plt.figure(figsize=(10, 5))

plt.bar(full2_yp['year'], full2_yp['price'])
plt.xticks(range(1970, 2070, 10))
plt.show()

# year별 price 합계 비교
full2_year_price2 = full2_data.groupby('year')['price'].sum()
full2_yp2 = pd.DataFrame(full2_year_price2)
full2_yp2.reset_index(drop = False, inplace=True)
full2_yp2.head()

# 2000년도부터 2020년까지의 값만 확인
condition4 = (full2_yp2['year']>=2000) & (full2_yp2['year']<2060)
full2_yp3 = full2_yp2.loc[condition4]

fig = plt.figure(figsize=(12, 4))

plt.bar(full2_yp3['year'], full2_yp3['price'])
plt.xticks(range(2000, 2021, 1))
plt.show()

# mileage 컬럼
print(full2_data['mileage'].describe())

# 구간별 데이터 확인하기
full2_data['mileage_cut'] = pd.cut(full2_data.mileage, bins=[0, 30000, 60000, 90000, 120000, 150000, 180000, 210000, 240000, 270000, 300000, 330000], labels=['30000 이하', '60000 이하', '90000 이하', '120000 이하', '150000 이하', '180000 이하', '210000 이하', '240000 이하', '270000 이하', '300000 이하', '300000 초과'])
full2_data['mileage_cut'].value_counts()

full2_mp = pd.DataFrame(full2_data.groupby('mileage_cut')['price'].sum())
full2_mp.reset_index(inplace=True)
full2_mp.head()

full2_m = pd.DataFrame(full2_data['mileage_cut'].value_counts())
full2_m.reset_index(inplace=True)
full2_m.head()

# 데이터프레임 병합
full2_m_mp = pd.merge(full2_m, full2_mp, on='mileage_cut', how='left')
full2_m_mp.head()
print(full2_m_mp)

# 이중 y축 시각화
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

x = full2_m_mp['mileage_cut']
y1 = full2_m_mp['count']
y2 = full2_m_mp['price']

fig, ax1 = plt.subplots()
ax1.set_xlabel('mileage range')
ax1.set_ylabel('count per range')
ax1.plot(x, y1, color='skyblue')

ax2 = ax1.twinx()
ax2.set_ylabel('sum price per range')
ax2.plot(x, y2, color='orange')

plt.show()
