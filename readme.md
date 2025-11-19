# 🎤 Whisper-Medium Fine-tuned for Korean ASR

한국어 음성 인식을 위해 OpenAI Whisper-Medium 모델을 파인튜닝한 프로젝트입니다.

## 📊 모델 정보

- **Base Model**: OpenAI Whisper-Medium
- **Language**: Korean
- **Sample Rate**: 16kHz
- **Training Strategy**: 2-Stage Fine-tuning
  - Stage 1: 대량 데이터 기본 학습
  - Stage 2: 타겟 도메인 적응

## 🎯 성능

| Metric | Value |
|--------|-------|
| CER    | X.XX% |
| WER    | X.XX% |

*검증 데이터 XXX개 샘플 기준*

## 🚀 빠른 시작

### 설치
```bash
# 저장소 클론
git clone https://github.com/yourusername/asr_certificate_2025.git
cd asr_certificate_2025

# 의존성 설치
pip install -r requirements.txt
```

### 추론 예시
```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# 모델 로드
model = WhisperForConditionalGeneration.from_pretrained("./model")
processor = WhisperProcessor.from_pretrained("./model")

# 오디오 로드
audio, sr = librosa.load("audio.wav", sr=16000)

# 추론
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
predicted_ids = model.generate(inputs["input_features"])
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print(transcription)
```

## 🧪 모델 검증

### 설정

`config/config.yml` 파일을 수정하여 검증 데이터 경로 설정:
```yaml
validation:
  data_dir: "./data/test/audio"
  manifest: "./data/test/manifest.json"
  model_path: "./model"
  output_dir: "./results"
```

### 실행
```bash
python validator.py
```

### 출력

- 엑셀 파일: `results/validate_YYYYMMDD_HHMMSS_{샘플수}.xlsx`
- 컬럼: 파일명, 정답, 예측, CER, 추론시간(초)

## 📁 프로젝트 구조
```
asr_certificate_2025/
├── model/                      # 파인튜닝된 모델 (Git LFS)
│   ├── model.safetensors      # 모델 가중치
│   ├── config.json
│   ├── tokenizer_config.json
│   └── ...
├── src/                        # 소스 코드
│   └── augmentor.py           # 음성 증강 모듈
├── config/                     # 설정 파일
│   └── config.yml             # 검증 설정
├── data/                       # 데이터 (로컬 전용)
│   └── test/                  # 테스트 데이터
├── validator.py                # 모델 검증 스크립트
├── requirements.txt            # 의존성
└── README.md
```

## 📦 데이터셋

### 학습 데이터
- 뉴스(35k), 커머스(13.5k), 강의/상담/전화/..(9.6k)
- 형식: MP3/WAV, 16kHz

*저작권 문제로 학습 데이터는 공개하지 않습니다.*

### Manifest 형식
```json
[
  {
    "audio": "audio_001.wav",
    "text": "전사된 텍스트"
  },
  {
    "audio": "audio_002.wav",
    "text": "두 번째 샘플"
  }
]
```

## 🔧 음성 증강

`src/augmentor.py`는 다음 증강 기법을 지원합니다:

- **Noise Addition**: 랜덤 노이즈 추가
- **Pitch Shift**: 음높이 변경
- **Speed Change**: 속도 조절
- **Gain**: 볼륨 조정
- **Low/High Pass Filter**: 주파수 필터링
- **Phase Inversion**: 위상 반전

**사용 예시:**
```python
from src.augmentor import HybridAudioAugmentor

augmentor = HybridAudioAugmentor(sr=16000)
results = augmentor.run_augmentation("input.wav", "./output_dir")
```