# 기본 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# CSV 파일을 데이터프레임으로 변환
df = pd.read_csv('auto-mpg.csv',header=None)
# 열 이름 지정
df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']
# 데이터 살펴보기
df.head( )

# 데이터 자료형 확인
df.info( )

# 데이터 통계 요약정보 확인
df.describe( )

# 누락 데이터 확인
df.isnull( ).sum( )

# 중복 데이터 확인
df.duplicated( ).sum( )

# 상관계수 분석 - 데이터프레임
corr = df.corr(numeric_only=True)
corr

# 상관계수 분석 - 히트맵
mask = np.triu(np.ones_like(corr,dtype=bool))

# 히트맵 그리기
plt.figure(figsize=(10,8))
sns.heatmap(corr,mask=mask,cmap='coolwarm',
            annot=True,fmt=".2f",cbar=True,square=True)
plt.show( )

# horsepower 열의 고유값 확인
df['horsepower'].unique( )

# horsepower 열의 자료형 변경(문자 -> 숫자)
df['horsepower'] = df['horsepower'].replace('?',np.nan)
df['horsepower'] = df['horsepower'].astype('float')
df.describe( )

# 결측치 제거
print(df['horsepower'].isnull().sum())
df_nan = df.dropna(subset=['horsepower'],axis=0)
print(df_nan['horsepower'].isnull().sum())

# 결측치 대체
print(df['horsepower'].isnull().sum())
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].mean())
print(df['horsepower'].isnull().sum())

# 상관계수 분석 - 데이터프레임
corr = df.corr(numeric_only=True)
corr

# 분석에 활용할 열 선택
ndf = df[['mpg','cylinders','horsepower','weight']]
ndf.head( )

# 종속 변수 Y인 "연비(mpg)"와 다른 변사 간의 선형관계를 그래프(산점도)로 확인
ndf.plot(kind='scatter',x='weight',y='mpg',c='coral',s=10,figsize=(10,5))
plt.show( )

# seaborn으로 산점도 그리기
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
sns.regplot(x='weight',y='mpg',data=ndf,ax=ax1)               # 회귀선 표시
sns.regplot(x='weight',y='mpg',data=ndf,ax=ax2,fit_reg=False) # 회귀선 미표시
plt.show( )

# seaborn 조인트 그래프 - 산점도, 히스토그램
sns.jointplot(x='weight',y='mpg',data=ndf)            # 회귀선 없음
sns.jointplot(x='weight',y='mpg',kind='reg',data=ndf) # 회귀선 표시

# 속성(변수) 선택
X = ndf[['weight']] # 독립 변수 X
y = ndf['mpg']      # 종속 변수 Y

# train data와 test data로 구분(7:3 비율)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)

print('train data 개수:',len(X_train))
print('test data 개수:',len(X_test))

# sklearn 라이브러리에서 선형회귀분석 모듈 가져오기
from sklearn.linear_model import LinearRegression

# 단순회귀분석 모델 객체 생성
lr = LinearRegression()

# train data를 가지고 모델 학습
lr.fit(X_train, y_train)

# 학습을 마친 모델에 test data를 적용하여 결정계수(R-제곱) 계산
r_square = lr.score(X_test,y_test)
print('R^2 결정계수:', r_square)

# 회귀식의 기울기
print('기울기 a:',lr.coef_)

# 회귀식의 y절편
print('y절편 b',lr.intercept_)

# 모델에 test date 데이터를 입력하여 예측한 값 y_hat을 실제 값 y와 비교
y_hat = lr.predict(X_test)

# 오차 계산
test_preds = pd.DataFrame(y_test)
test_preds.columns = ['y_test']
test_preds['y_hat'] = y_hat
test_preds['squared_error'] = (test_preds['y_hat'] - test_preds['y_test'])**2
test_preds

# 평균 제곱 오차
mse = test_preds['squared_error'].mean()
print('mse:',mse)

# 오차 분석
fig, axes = plt.subplots(1,2,figsize=(10,5))
sns.regplot(x='y_test',y='y_hat',data=test_preds,ax=axes[0]);
sns.kdeplot(x='squared_error',data=test_preds,ax=axes[1]);







