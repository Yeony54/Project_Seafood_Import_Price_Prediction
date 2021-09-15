# 수산물 수입가격 예측을 통한 최적의 가격 예측 모형 도출
*2021 BigContest Championship League*

## 1. 파일 소개
*1. 제공데이터 EDA*: 제공데이터를 분석하여, 분석 방향을 도출한 파일입니다  

*2. 추가데이터 EDA* : 제공데이터외에 자료조사를 하여 필요하다고 여긴 세계날씨(강수량, 풍속, 기온), 한국 날씨(강수량, 수온, 기온), CPI지수, 유가, 환율 데이터를 EDA한 파일입니다.   

*3.0추가 데이터 전처리*: 유가, 한국 날씨, CPI지수 데이터를 전처리한 파일입니다. 

*3.1.전처리 가설 검증_전체국가* : 기존 제공데이터와 세계날씨와 유가를 제외한 추가 데이터로 가설을 세운 다음, 파생변수 추가, 상관관계 분석 및 시각화를 통해 가설을 검증한 파일입니다.  

*3.2.전처리 가설 검증_주요 국가 중심* : 기존 제공데이터와 추가데이터 전부를 사용하여 주요국가(칠레, 중국, 노르웨이, 태국, 베트남)와 주요 수출국별 주요 어종(오징어, 연어, 흰다리새우)만을 통해 파생변수 추가, 상관관계 분석 및 시각화를 통해 가설을 검증한 파일입니다. 

*4.1.*:

*4.2.시계열 모델.ipynb* : 주요국가(칠레, 중국, 노르웨이, 태국, 베트남) 주차별 상세어종(오징어, 연어, 흰다리새우) 가격 평균을 통해 prophet, RNN, LSTM, GRU를 통해 rmse측정 및 가격 예측을 한 파일입니다.  


## Links
- [빅콘테스트](https://www.bigcontest.or.kr/index.php)
- [주요 농산물 가격예측 시스템](https://www.gyeongnam.go.kr/bigdatafarm/index.es?sid=a1#close)

## DATA
- [빅콘테스트 2021 수산물 수입가격 예측을 통한 최적의 가격 예측 모형 도출](https://www.bigcontest.or.kr/)
- [수심별 염도](http://www.climate.go.kr/home/09_monitoring/marine/salt_avg)
- [지출목적별 소비자물가지수, 소비자물가지수](https://kosis.kr/index/index.do)
- [환율,원유](https://kr.investing.com/currencies/cny-krw-historical-data)

## References
### Studies
- [계절성을 고려한 가공유형별 오징어 소매가격 예측모형 비교 분석](http://english.ksfme.or.kr/xml/15446/15446.pdf)
- 제주 양식넙치의 월별 산지가격 예측 및 예측력 비교
- 시계열 분석을 이용한 굴 가격 예측에 관한 연구
- [VAR 모형을 이용한 유통단계별 갈치가격의 인과성 분석](https://scienceon.kisti.re.kr/commons/util/originalView.do?cn=JAKO201516351715460&oCn=JAKO201516351715460&dbt=JAKO&journal=NJOU00293779)
- [FTA 체결 이후 수입수산물의 유통·소비 현황과 과제](https://www.nkis.re.kr:4445/subject_view1.do?otpId=KMI00053255&otpSeq=0&popup=P)
- [환율과 환율변동성의 변화가 수산물 수입에 미치는 영향분석](https://kiss.kstudy.com/thesis/thesis-view.asp?key=2565449)
### Kaggle
- [자전거 수요 예측](https://www.kaggle.com/viveksrinivasan/eda-ensemble-model-top-10-percentile)
- [집값 예측](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python)
### Etc
- KAPI·Korea Agricultural product Price Index
- [세계환율](https://kr.investing.com/currencies/cny-krw-historical-data)

