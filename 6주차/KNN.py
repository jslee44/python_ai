import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# seaborn에서 제공하는 titanic 데이터셋 가져오기
df = sns.load_dataset('titanic')
df.head( )

# 데이터 자료형 확인
df.info( )

# 데이터 통계 요약정보 확인
df.describe( )

# 데이터 통계 요약정보 확인(범주형)
df.describe(include='object')

# 누락 데이터 확인
df.isnull( ).sum( )

# 중복 데이터 확인
df.duplicated( ).sum( )

# 목표 변수
df['survived'].value_counts( )

# 목표변수 시각화
sns.countplot(data=df, x='survived');

# 시각화
g = sns.FacetGrid(df,col='survived',row='pclass',hue='sex')
g.map(sns.kdeplot,'age',alpha=0.5,fill=True)
g.add_legend( );

# 형제자매 및 배우자의 수에 대한 시각화
sns.displot(x='sibsp',kind='hist',hue='survived',data=df,multiple='fill');

# 부모 및 자식의 수에 대한 시각화
sns.displot(x='parch',kind='hist',hue='survived',data=df,multiple='fill');

# 탑승 항구와 나이에 대한 생존 여부 시각화
sns.boxplot(x='embarked',y='age',hue='survived',data=df)

# 중복 데이터 제거
print('중복 제거 이전:',df.shape)
df = df.drop_duplicates()
print('중복 제거 이후:',df.shape)

# 데이터가 존재하지 않는 열 및 중복 열 삭제
rdf = df.drop(['deck','embark_town'],axis=1)
rdf.columns.values

# age 열에 나이 데이터가 없는 모든 행 삭제
rdf = rdf.dropna(subset=['age'],how='any',axis=0)
print(len(rdf))

# embarked 열의 NaN값을 승선 도시 중에서 가장 많이 출현한 값 찾기
most_freq = rdf['embarked'].value_counts(dropna=True).idxmax()
print(most_freq)

# 최빈값으로 누락 데이터 치환하기
rdf['embarked'] = rdf['embarked'].fillna(most_freq)

# 결측치 확인하기
rdf.isnull().sum()

# 분석에 활용할 열(특성) 선택
ndf = rdf[['survived','pclass','sex','age','sibsp','parch','embarked']]
ndf.head( )

# 원핫인코딩(one-hot-encoding) - 범주형 데이터를 모형이 인식할 수 있도록 숫자형으로 변환
onehot_sex = pd.get_dummies(ndf['sex'])
ndf = pd.concat([ndf,onehot_sex],axis=1)

onehot_embarked = pd.get_dummies(ndf['embarked'],prefix='town')
ndf = pd.concat([ndf,onehot_embarked],axis=1)

ndf = ndf.drop(['sex','embarked'],axis=1)
ndf.head()

# 속성(변수) 선택
x = ndf[['pclass','age','sibsp','parch','female',
         'male','town_C','town_Q','town_S']]
y = ndf['survived']

# 설명 변수 데이터를 정규화
from sklearn import preprocessing
x = preprocessing.StandardScaler().fit(x).transform(x)

# train data와 test data로 구분(7:3 비율)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=10)

print('train data 개수:',x_train.shape)
print('test data 개수:', x_test.shape)

# sklearn 라이브러리에서 KNN 분류 모형 가져오기
from sklearn.neighbors import KNeighborsClassifier

# 모형 객체 생성(k=5로 설정)
knn = KNeighborsClassifier(n_neighbors=5)

# train data를 가지고 모형 학습
knn.fit(x_train,y_train)

# test data를 가지고 y_hat을 예측(분류)
y_hat = knn.predict(x_test)

print('예측값:', y_hat[0:10])
print('실제값:', y_test.values[0:10])

# 모형 성능 평가
from sklearn import metrics
knn_matrix = metrics.confusion_matrix(y_test,y_hat)
print(knn_matrix)

# Confusion Matrix 시각화
plt. figure(figsize=(8,6))
sns.heatmap(knn_matrix,annot=True,fmt='d',cmap='Blues',
            xticklabels=['Negative','Positive'],
            yticklabels=['Negative','Positive'])
plt.title('Confusion Matrix')
plt.ylabel('Actual label')
plt.show( )









