# 다국어 영수증 OCR
**2024.10.30 - 2024.11.07**  
**학습 데이터 추가와 수정을 통한 Data-Centric 다국어 영수증 속 글자 검출**  
**Naver Connect & Upstage 대회**

---

## Authors
- [민창기](https://github.com/min000914)  
- [이준학](https://github.com/danlee0113)  
- [김민환](https://github.com/alsghks1066)  
- [김준원](https://github.com/KimJunWon98)  
- [전인석](https://github.com/inDseok)  
- [신석준](https://github.com/SeokjunShin)  

---

## 대회 소개
OCR (Optical Character Recognition) 기술은 이미지 속 문자를 컴퓨터가 인식할 수 있도록 변환하는 기술입니다.  
본 대회는 다국어(중국어, 일본어, 태국어, 베트남어)로 작성된 영수증 이미지에서 글자 검출을 수행하는 Task를 목표로 합니다.  

- **글자 검출(Task)**: 이미지에서 글자가 있는 위치를 예측하는 모델 개발
- **언어 지원**: 중국어, 일본어, 태국어, 베트남어
- **특징**: 글자 검출에만 초점, 글자 인식 및 정렬은 제외

---

## 평가 방법
DetEval 방식으로 모델 성능을 평가합니다.  
다중 매칭을 허용하며, 박스 레벨에서 정답 여부를 판단합니다. 주요 조건은 다음과 같습니다:  

1. **매칭 조건**  
   - **One-to-One**: 정답 박스 1개와 예측 박스 1개가 매칭되며 조건 성립  
   - **One-to-Many**: 정답 박스 1개와 예측 박스 여러 개가 매칭되며 패널티 적용  
   - **Many-to-One**: 정답 박스 여러 개와 예측 박스 1개가 매칭되며 패널티 적용  

2. **기본 조건**  
   - Area Recall ≥ 0.8  
   - Area Precision ≥ 0.4  

3. **평가지표**  
   - **Precision**: 예측 박스 중 올바른 비율  
   - **Recall**: 정답 박스를 올바르게 검출한 비율  
   - **F1 Score**: Precision과 Recall의 조화 평균  

---

## 리더보드
| Dataset  | Precision | Recall | F1 Score |
|----------|-----------|--------|----------|
| Public   | 0.9327    | 0.8657 | 0.8980   |
| Private  | 0.9187    | 0.8434 | 0.8795   |

---

## Dataset
- **Train**: 중국어, 일본어, 태국어, 베트남어 영수증 각 100장씩 (총 400장)  
- **Test**: 동일 언어 영수증 각 30장씩 (총 120장)  
- **출처**: 부스트캠프 AI Tech  

---

## 개발 환경
- **GPU**: v100  

---

## 데이터 클리닝
400장의 이미지 중 286장의 라벨링 오류를 수정했습니다.  
- **툴**: [CVAT](https://www.cvat.ai/)  
- **기준**:  
  1. 글자 형태에 맞춰 점 4개로 라벨링  
  2. 글자 표현이 어려운 경우 끊어서 라벨링  

### 예시
- **클리닝 전**  
  ![클리닝 전](https://github.com/user-attachments/assets/47f73f9a-78f7-4597-8119-ff342c756395)  

- **클리닝 후**  
  ![클리닝 후](https://github.com/user-attachments/assets/c54c130f-1b01-4657-a721-b91ced474200)  

---

## 크롭 앙상블
이미지를 슬라이딩 크롭하여 추론 후, 결과를 앙상블합니다.  

1. **원본 이미지 추론**  
   ![원본 이미지](https://github.com/user-attachments/assets/0d74a046-dd83-439d-8940-6a0ac3ef1adf)  

2. **크롭 이미지 추론**  
   - ![Crop 1](https://github.com/user-attachments/assets/193527de-e44d-481c-9e88-dd432e7a1c52)  
   - ![Crop 2](https://github.com/user-attachments/assets/bcc699ef-95e6-476b-8893-7cee0b753eec)  
   - ![Crop 3](https://github.com/user-attachments/assets/b3e79bd6-7971-4684-8908-4d3afa2f5170)  
   - ![Crop 4](https://github.com/user-attachments/assets/15cbbe73-b8c2-4055-aa29-86c94cbfa9a6)  

3. **최종 앙상블**: 모든 결과를 합산  

---

## 추가 데이터셋
1. **CORD**: 사용하지 않음 (블러 처리 및 라벨링 미비)  
2. **SROIE2019**: 영어 영수증 데이터 (총 973장)  
   - ![SROIE2019 예시](https://github.com/user-attachments/assets/ba7ea5ab-ecfa-4a86-84f7-db160bb12edc)  

---

## 데이터 증강
증강 기법으로 데이터셋을 2배 확장하여 모델 학습에 활용했습니다.  

### 증강 전후 비교
- **원본 데이터**  
  ![원본 데이터](https://github.com/user-attachments/assets/3687e95f-6b17-413e-8f4b-77b401e5406d)  

- **증강 데이터**  
  ![증강 데이터](https://github.com/user-attachments/assets/e19de2fa-7fd8-4ac3-981f-984971e4e462)  

### 주요 증강 기법
- `HueSaturationValue`  
- `MotionBlur`  
- `CLAHE`  
- `RandomPolygonShadow` (커스텀)  
- `RGBShift`  

 RandomPolygonShadow는 커스텀 증강으로 데이터에 그림자를 표현하기 위해 투명도가 조절되는 랜덤 폴리곤 도형을 이미지에 Plot하는 증강입니다.

---

## 실험 결과
| IoU & Vote | Precision | Recall  | F1 Score |
|------------|-----------|---------|----------|
| 0.5, 9     | 0.9506    | 0.7820  | 0.8581   |
| 0.4, 6     | 0.9187    | 0.8434  | 0.8795   |
| 0.35, 7    | 0.9180    | 0.8416  | 0.8781   |
| 0.2, 4     | 0.8666    | 0.8187  | 0.8420   |

---
