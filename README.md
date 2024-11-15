
# 다국어 영수증 OCR
- 2024.10.30 - 2024.11.07 
- 학습 데이터 추가와 수정을 통한 Data-Centric 다국어 영수증 속 글자 검출
- Naver Connect & Upstage 대회 

## Authors

- [민창기](https://github.com/min000914)  [이준학](https://github.com/danlee0113)  [김민환](https://github.com/alsghks1066)  [김준원](https://github.com/KimJunWon98)  [전인석](https://github.com/inDseok)  [신석준](https://github.com/SeokjunShin)



## 대회 소개
카메라로 영수증을 인식할 경우 자동으로 영수증 내용이 입력되는 어플리케이션이 있습니다. 이처럼 OCR (Optical Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다.

OCR은 글자 검출 (text detection), 글자 인식 (text recognition), 정렬기 (Serializer) 등의 모듈로 이루어져 있습니다. 본 대회는 아래와 같은 특징과 제약 사항이 있습니다.

본 대회에서는 다국어 (중국어, 일본어, 태국어, 베트남어)로 작성된 영수증 이미지에 대한 OCR task를 수행합니다.

본 대회에서는 글자 검출만을 수행합니다. 즉, 이미지에서 어떤 위치에 글자가 있는지를 예측하는 모델을 제작합니다.

대회 기간과 task 난이도를 고려하여 코드 작성에 제약사항이 있습니다. 상세 내용은 Data > Baseline Code (베이스라인 코드)에 기술되어 있습니다.


## 평가 방법

모델의 성능은 DetEval 방식으로 평가됩니다. DetEval은 이미지 레벨에서 정답 박스가 여러개 존재하고, 예측한 박스가 여러개가 있을 경우, 박스끼리의 다중 매칭을 허용하여 점수를 주는 평가방법입니다.

모든 정답 박스와 예측 박스를 순회하면서, 매칭이 되었는지 판단하여 박스 레벨로 정답 여부를 측정합니다. 박스들이 매칭이 되는 조건은 박스들을 순회하면서, 계산한 Area Recall, Area Precision이 0 이상일 경우 매칭 여부를 판단하게 되며, 박스의 정답 여부는 Area Recall 0.8 이상, Area Precision 0.4 이상을 기준으로 하고 있습니다.

매칭이 되었는가 대한 조건은 크게 3가지 조건이 있습니다.

- one-to-one match: 정답 박스 1개와 예측 박스 1개가 매칭 && 기본조건 성립

- one-to-many match: 정답 박스 1개와 예측 박스 여러개가 매칭되는 경우. 이 조건에서는 박스 recall / precision 에 0.8 로 penalty가 적용됩니다.

- many-to-one match: 정답 박스 여러개와 예측박스 1개가 매칭되는 경우


## 리더보드 
- Public: 0.9327	0.8657	0.8980
- Private: 0.9187	0.8434	0.8795

 위의 점수는 순서대로 precision, recall, f1 score을 의미합니다.
## Dataset

- train: 태국어, 베트남어, 일본어, 중국어 영수증 데이터 각 100장씩, 총 400장의 이미지
- test: 동일한 언어의 영수증 데이터 각 30장씩, 총 120장의 이미지
- 출처: 부스트캠프 AI Tech 

## 개발 환경

- GPU : v100


## 추가해야 할 내용

데이터 클리닝 작업(무슨작업, 어떤 가이드라인)
작업사진전 -> 후 비교

추가 데이터셋 확보
CORD, SROIE2019
각 데이터셋 사진

CORD를 쓰지 않은 이유

원본+증강 데이터 사용
증강데이터사진

FOLD별로 다양한 Size, 다양한 Crop을 적용하여 훈련

앙상블설명



![original_image](https://github.com/user-attachments/assets/0d74a046-dd83-439d-8940-6a0ac3ef1adf)
![2_3_crop_row1_col1](https://github.com/user-attachments/assets/193527de-e44d-481c-9e88-dd432e7a1c52)
![2_3_crop_row1_col2](https://github.com/user-attachments/assets/bcc699ef-95e6-476b-8893-7cee0b753eec)
![2_3_crop_row2_col1](https://github.com/user-attachments/assets/b3e79bd6-7971-4684-8908-4d3afa2f5170)
![2_3_crop_row2_col2](https://github.com/user-attachments/assets/15cbbe73-b8c2-4055-aa29-86c94cbfa9a6)

크롭앙상블 설명



------------------------------------------------------------------------------------------------------



![X51006008073](https://github.com/user-attachments/assets/3687e95f-6b17-413e-8f4b-77b401e5406d)

원본 Train 데이터


------------------------------------------------------------------------------------------------------



![aug_X51006008073](https://github.com/user-attachments/assets/e19de2fa-7fd8-4ac3-981f-984971e4e462)

aug적용 데이터


------------------------------------------------------------------------------------------------------



![cord2_test_image_17](https://github.com/user-attachments/assets/a0be4b86-18d3-4cdb-9297-a3b043c0a9e3)

cord2 안쓴 이유


------------------------------------------------------------------------------------------------------



![X51005447840](https://github.com/user-attachments/assets/ba7ea5ab-ecfa-4a86-84f7-db160bb12edc)

SROIE2019 데이터 Plot


------------------------------------------------------------------------------------------------------



![extractor zh in_house appen_000314_page0001](https://github.com/user-attachments/assets/47f73f9a-78f7-4597-8119-ff342c756395)

클리닝 전


------------------------------------------------------------------------------------------------------



![extractor zh in_house appen_000314_page0001](https://github.com/user-attachments/assets/c54c130f-1b01-4657-a721-b91ced474200)

클리닝 후


------------------------------------------------------------------------------------------------------



![japan000215_resultPlot](https://github.com/user-attachments/assets/0d51e3bf-f30e-493c-b84f-fb885f364ebe)

앙상블 전 결과 이미지 Plot



------------------------------------------------------------------------------------------------------



![japan000215_esemble_resultPlot](https://github.com/user-attachments/assets/9593a91b-f08b-4bf2-9db0-b872974d0b28)

앙상블 후 결과 이미지 Plot


------------------------------------------------------------------------------------------------------



![annotated_extractor th in_house appen_000692_page0001](https://github.com/user-attachments/assets/0a111def-7201-4f88-aaa5-77466ae6db2a)
![image](https://github.com/user-attachments/assets/9db1f5a7-4319-4ee0-838d-e1fbf5bf3795)

Rotate 미적용


------------------------------------------------------------------------------------------------------


![image](https://github.com/user-attachments/assets/91c3e9f4-adad-4a96-949e-0a55b95358f6)
![image](https://github.com/user-attachments/assets/bf281c3b-f7e3-45d1-b752-7ad4981e688a)

Rotate 적용








