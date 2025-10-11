# KcBERT-MedicalSentiment-Classification

2024년 의사 집단행동과 관련된 네이버 뉴스 댓글을 기반으로 구축된  
**KcBERT 기반 의사 감정 분류 모델**입니다.  

본 모델은 온라인 공간에서 드러난 대중의 ‘의사’ 집단에 대한 정서를  
**중립**, **분노**, **혐오** 로 구분하여 분석하는 것을 목표로 합니다.

---

## 1. Dataset Description

| 컬럼명 | 설명 |
|--------|------|
| comments | 뉴스 댓글 원문 |
| target | 1 = 의사 대상, 0 = 비의사 |
| label | 0 = 중립, 1 = 분노, 2 = 혐오 |

초기 약 2,000건의 수작업 라벨링 데이터를 구축하였으며,  
상대적으로 데이터가 부족한 **중립** 및 **분노** 범주를 중심으로 데이터 증강을 수행했습니다.  
- **중립 댓글 436건**:  
  - 100건은 의사 집단행동 관련 뉴스 본문에서 의사 관련 문장을 발췌하여 직접 구성  
  - 336건은 규칙 기반 문장 생성(rule-based sentence generation)으로 보강  
- **분노 댓글 500건**: 규칙 기반 문장 생성으로 증강  
- **오분류 데이터 100건**: 모델 검증 과정에서 잘못 분류된 사례를 재라벨링하여 추가 반영  

최종적으로 **총 3,036건의 댓글 데이터**를 확보하여 KcBERT-base 모델을 파인튜닝했습니다.



### 1)정서 개념 설명
| 정서 | 개념적 정의 |
|------|-------------|
| **중립 (Neutral)** | 특정 행위나 집단에 대한 분노 또는 혐오 정서를 내포하지 않는 상태.  부당함에 대한 가치판단이 유보되거나, 정책적·사실적 정보 전달, 단순 의견 제시에 해당.  감정의 방향성이 분명하지 않거나 맥락이 모호한 표현도 포함. |
| **분노 (Anger)** | 부당함에 대한 합리적 판단에 기반하여 문제를 개선하고자 하는 **미래지향적·실천적 감정**.  ‘응보의 중단’과 ‘실천으로서의 분노’ 개념을 따르며, 자기 존엄의 회복과 사회적 부정에 대한 **도덕적 저항의 정념**으로 정의.  복수가 아닌 개선을 지향하는 관용적이고 일시적인 정서. |
| **혐오 (Hate)** | 부당함에 대한 합리적 판단이 결여된 채 대상의 **도덕적·존재적 파괴를 전제**하는 감정.  타자에 대한 배제, 비인간화, 낙인 등 파괴적 권력 감정을 포함하며,  사랑·관용·이해의 가능성을 차단하는 **파괴적 분노**로 정의.  즉, 감정이 아닌 **사회문화적 권력의 발현**으로 간주. |

### 2)라벨링 예시 

| comments | target | label |
|-----------|---------|--------|
| 국민이 위협을 느끼는 것은 의사들이 아니고 윤통이다. | 0 |  |
| 파업 지지 합니다! 이 정책 통과되면 앞으로 국민들 의료비 폭등합니다ㅠㅠ 미국처럼 맹장수술 하나 받는데 몇백만원 내야돼요 제발 다들 정신 좀 차리세요 국민 여러분들ㅜㅠㅠ | 1 | 0 |
| "의사를 존중하는 이유는 성공한 엘리트집단이라서가 아니라 직업윤리상 헌신이 기저에 깔려있기 때문이다. 이걸 내팽겨버린 지금 니들편은 니들밖에 없다 | 1 | 1 |
| 의사들만 따로 무인도에 모여 살게하면 좋을듯. 남들은 어떻게되든 자기 배만 까~득~ 채우려는 심보. | 1 | 2 |

---

## 2. Model Pipeline

1️ **의사 대상 분류 모델 (Target Classification)**  
   → 해당 댓글이 ‘의사’를 직접적으로 표적으로 하는지 판단  

2️⃣ **의사 감성 분류 모델 (Emotion Classification)**  
   → 타겟=1인 댓글 중에서 감정(중립·분노·혐오)을 분류  

---

## 3. Model Performance

### 의사 대상 분류 모델 (Target Classification)

| 라벨 | 정밀도 | 재현율 | F1-점수 |
|------|--------|--------|--------|
| 비의사 | 0.88 | 0.88 | 0.88 |
| 의사 | 0.92 | 0.92 | 0.92 |
| **정확도(Accuracy)** |  |  | **0.90** |

---

### 의사 감성 분류 모델 (Emotion Classification)

| 라벨 | 정밀도 | 재현율 | F1-점수 |
|------|--------|--------|--------|
| 중립 | 0.89 | 0.90 | 0.90 |
| 분노 | 0.90 | 0.86 | 0.88 |
| 혐오 | 0.87 | 0.89 | 0.88 |
| **정확도(Accuracy)** |  |  | **0.88** |



### Hugging Face 모델
- **의사 대상 분류 모델:** [[JunHyeongdd/doctor_target_ko](https://huggingface.co/JunHyeongdd/doctortargetmodel)]
- **의사 감성 분류 모델:** [[JunHyeongdd/doctor_emotion_ko](https://huggingface.co/JunHyeongdd/doctorsentimentmodel)]

---

## 4. How to install and run

```python

from transformers import pipeline

# 1️⃣ 의사 대상 분류 모델 (Target Classification)
pipe_target = pipeline("text-classification", model="JunHyeongdd/doctortargetmodel")

# 2️⃣ 의사 감성 분류 모델 (Emotion Classification)
pipe_sentiment = pipeline("text-classification", model="JunHyeongdd/doctorsentimentmodel")

# 결과 해석용 라벨 매핑
target_labels = {
    "LABEL_0": "비의사",
    "LABEL_1": "의사"
}

sentiment_labels = {
    "LABEL_0": "중립",
    "LABEL_1": "분노",
    "LABEL_2": "혐오"
}

# ------------------------------------------
# 댓글 입력
# ------------------------------------------
comment = input("댓글을 입력하세요: ")

# ------------------------------------------
# 1단계: 의사 대상 분류 실행
# ------------------------------------------
result_target = pipe_target(comment)[0]
target_label = target_labels[result_target['label']]
target_score = result_target['score']

# ------------------------------------------
# 2단계: 의사 감성 분류 실행 (의사 대상일 때만)
# ------------------------------------------
if target_label == "의사":
    result_sentiment = pipe_sentiment(comment)[0]
    sentiment_label = sentiment_labels[result_sentiment['label']]
    sentiment_score = result_sentiment['score']

    print(f"\n[입력 댓글] {comment}")
    print(f"▶ 대상 분류 결과: {target_label} ({target_score:.3f})")
    print(f"▶ 감정 분류 결과: {sentiment_label} ({sentiment_score:.3f})")

else:
    print(f"\n[입력 댓글] {comment}")
    print(f"▶ 대상 분류 결과: {target_label} ({target_score:.3f})")
    print("▶ 감정 분류는 생략되었습니다. (비의사 관련 댓글)")
