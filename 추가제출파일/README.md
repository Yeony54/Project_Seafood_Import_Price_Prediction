# 수산물 수입가격 예측을 통한 최적의 가격 예측 모형 도출
*2021 BigContest Championship League*

## 1. 파일 소개
*1. 제공데이터 EDA*: 제공데이터를 분석하여, 분석 방향을 도출한 파일입니다  

*2. 추가데이터 EDA* : 제공데이터외에 자료조사를 하여 필요하다고 여긴 세계날씨(강수량, 풍속, 기온), 한국 날씨(강수량, 수온, 기온), CPI지수, 유가, 환율 데이터를 EDA한 파일입니다.   

*3-0추가 데이터 전처리*: 유가, 한국 날씨, CPI지수 데이터를 전처리한 파일입니다. 

*3-1.전처리 가설 검증_전체국가* : 기존 제공데이터와 세계날씨와 유가를 제외한 추가 데이터로 가설을 세운 다음, 파생변수 추가, 상관관계 분석 및 시각화를 통해 가설을 검증한 파일입니다.  

*3-2.전처리 가설 검증_주요 국가 중심* : 기존 제공데이터와 추가데이터 전부를 사용하여 주요국가(칠레, 중국, 노르웨이, 태국, 베트남)와 주요 수출국별 주요 어종(오징어, 연어, 흰다리새우)만을 통해 파생변수 추가, 상관관계 분석 및 시각화를 통해 가설을 검증한 파일입니다. 

*4-1.모델 최적화*: 3-1, 3-2 가설 검증을 바탕으로 선정한 dataframe을 LinearRegression, Lasso, Ridge, ElasticNet, GradientBoostingRegressor, XGBRegressor, LGBMRegressor. RandomForest, DecisionTreeRegressor으로 RandomizedSearchCV를 통해 최적 params 도출 후, 최적 params로 평가데이터를 예측한 파일입니다.

*4.2.시계열 모델* : 주요국가(칠레, 중국, 노르웨이, 태국, 베트남) 주차별 상세어종(오징어, 연어, 흰다리새우) 가격 평균을 통해 prophet, RNN, LSTM, GRU를 통해 rmse측정 및 가격 예측을 한 파일입니다.  

*5.수산물 수입 가격 예측* : 최종 평가데이터의 수입가격을 예측합니다. 평가데이터 전처리부터 모델 학습, 그리고 최종 예측까지 진행합니다  

*utility.py* : 전처리에 쓰이는 함수 및 가설 검증에 쓰이는 함수입니다.


## 2. 추가 DATA
- [빅콘테스트 2021 수산물 수입가격 예측을 통한 최적의 가격 예측 모형 도출](https://www.bigcontest.or.kr/)
- [수심별 염도](http://www.climate.go.kr/home/09_monitoring/marine/salt_avg)
- [지출목적별 소비자물가지수, 소비자물가지수](https://kosis.kr/index/index.do)
- [환율,원유](https://kr.investing.com/currencies/cny-krw-historical-data)

## Links
- [빅콘테스트](https://www.bigcontest.or.kr/index.php)
- [주요 농산물 가격예측 시스템](https://www.gyeongnam.go.kr/bigdatafarm/index.es?sid=a1#close)
