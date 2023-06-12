import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance
import geopandas as gpd
import matplotlib.pyplot as plt
import sys

plt.rc('font', family='Malgun Gothic')

# 데이터 불러오기
df = pd.read_csv('terrain_data.csv')

# 가중치 정의
def calculate_weights(row, coords, pop_weight, car_weight, elderly_rate_weight, gender_ratio_weight):
    hospital_distance = distance.euclidean([row['Latitude'], row['Longitude']], coords)
    population_score = (row['Population'] / df['Population'].max() * pop_weight)**3
    car_count_score = (row['Number_of_Cars'] / df['Number_of_Cars'].max() * car_weight)**3
    elderly_population_score = (row['Elderly_Population_Rate'] / df['Elderly_Population_Rate'].max() * elderly_rate_weight)**3
    gender_ratio_score = (row['Gender_Ratio'] / df['Gender_Ratio'].max() * gender_ratio_weight)**3
    return hospital_distance / (population_score + car_count_score + elderly_population_score + gender_ratio_score)

# 병원 위치 결정 함수 정의
def choose_location(coords, pop_weight, car_weight, elderly_rate_weight, gender_ratio_weight):
    df['weight'] = df.apply(calculate_weights, axis=1, coords=coords, pop_weight=pop_weight, car_weight=car_weight, elderly_rate_weight=elderly_rate_weight, gender_ratio_weight=gender_ratio_weight)
    return df['weight'].sum()

# 가중치 세트 정의
a = float(sys.argv[1])

weight_sets = [
    (1.0, 1.0, 1.0, 1.0),
    (a, 1.0, 1.0, 1.0),
    (1.0, a, 1.0, 1.0),
    (1.0, 1.0, a, 1.0),
    (1.0, 1.0, 1.0, a),
]

# 초기 위치 정의
initial_coords = [35, 127]

# 결과 계산
results = []
for weights in weight_sets:
    pop_weight, car_weight, elderly_rate_weight, gender_ratio_weight = weights
    result = minimize(choose_location, initial_coords, args=(pop_weight, car_weight, elderly_rate_weight, gender_ratio_weight), method='L-BFGS-B')
    results.append(result.x)

# 시각화
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, color='red', zorder=5)

labels = ['모든 가중치 동일', '인구 가중치 증가', '자동차 수 가중치 증가', '노인 비율 가중치 증가', '성비 비율 가중치 증가']

# 최적 위치 그리기
for i, coords in enumerate(results):
    plt.scatter(coords[1], coords[0], zorder=5, label=labels[i])

# 최적 위치 출력
for i, result in enumerate(results):
    print(f"{labels[i]}에 대한 최적의 병원 위치: {result[0]}, {result[1]}")

# 각 점에 대해 지역 이름 추가
for x, y, label in zip(df['Longitude'], df['Latitude'], df['Location']):
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center')

plt.legend()
plt.show()
