"""
ASR 모델 검증 스크립트
- config/config.yml에서 설정 로드
- 검증 데이터에 대해 추론 수행
- 예측 문장, 정답 문장, CER, 추론 시간을 엑셀로 출력
"""
import os
import json
import time
import yaml
from datetime import datetime

import torch
import librosa
import pandas as pd
import evaluate
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def load_config(config_path="./config/config.yml"):
    """config.yml 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_manifest(manifest_path):
    """
    Manifest 파일 로드
    
    지원 형식:
    1. JSON 배열: [{"audio": "path", "text": "..."}, ...]
    2. JSONL: {"audio": "path", "text": "..."}\n{"audio": ...}\n
    """
    with open(manifest_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # JSON 배열 형식
    if content.startswith('['):
        return json.loads(content)
    
    # JSONL 형식
    else:
        manifest = []
        for line in content.split('\n'):
            if line.strip():
                manifest.append(json.loads(line))
        return manifest


def inference_single(audio_path, reference_text, model, processor, device):
    """
    단일 오디오 파일 추론
    
    Returns:
        dict: {
            'prediction': str,
            'reference': str,
            'cer': float,
            'inference_time': float (초)
        }
    """
    try:
        # 오디오 로드
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # 추론 시작
        start_time = time.time()
        
        with torch.no_grad():
            # Feature extraction
            input_features = processor.feature_extractor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features
            
            input_features = input_features.to(device)
            
            # Generate
            predicted_ids = model.generate(input_features, max_length=128)
            
            # Decode
            prediction = processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
        
        inference_time = time.time() - start_time
        
        # CER 계산
        cer_metric = evaluate.load("cer")
        cer = cer_metric.compute(
            predictions=[prediction], 
            references=[reference_text]
        )
        
        return {
            'prediction': prediction,
            'reference': reference_text,
            'cer': cer,
            'inference_time': inference_time
        }
        
    except Exception as e:
        print(f"  ⚠️  오류 발생: {audio_path} - {str(e)}")
        return {
            'prediction': f"[ERROR: {str(e)}]",
            'reference': reference_text,
            'cer': None,
            'inference_time': None
        }


def main():
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Config 로드
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    config = load_config("./config/config.yml")
    
    val_data_dir = config['validation']['data_dir']
    val_manifest = config['validation']['manifest_dir']
    model_path = config['validation']['model_path']
    output_dir = config['validation']['output_dir']
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("🔍 ASR 모델 검증")
    print("=" * 70)
    print(f"📦 모델 경로:     {model_path}")
    print(f"📊 Manifest:      {val_manifest}")
    print(f"📁 데이터 경로:   {val_data_dir}")
    print(f"💾 출력 디렉토리: {output_dir}")
    print(f"🖥️  디바이스:      {device}")
    print("=" * 70)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 모델 로드
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n📦 모델 로딩 중...")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print("✅ 모델 로드 완료!")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Manifest 로드
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print(f"\n📂 Manifest 로딩 중...")
    manifest = load_manifest(val_manifest)
    print(f"✅ 총 {len(manifest):,}개 샘플 로드됨")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 추론 수행
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print(f"\n🚀 추론 시작...\n")
    
    results = []
    total_start = time.time()
    
    for idx, item in enumerate(manifest, start=1):
        # 파일 경로
        audio_file = item.get('audio') or item.get('audio_filepath')
        audio_path = os.path.join(val_data_dir, audio_file)
        reference = item['text']
        
        # 추론
        result = inference_single(audio_path, reference, model, processor, device)
        
        # 결과 저장
        results.append({
            '파일명': audio_file,
            '정답': result['reference'],
            '예측': result['prediction'],
            'CER': result['cer'],
            '추론시간(초)': result['inference_time']
        })
        
        # 진행상황 출력 (100개마다)
        if idx % 100 == 0:
            valid_cers = [r['CER'] for r in results if r['CER'] is not None]
            avg_cer = sum(valid_cers) / len(valid_cers) if valid_cers else 0
            elapsed = time.time() - total_start
            eta = elapsed / idx * (len(manifest) - idx)
            
            print(f"진행: {idx:,}/{len(manifest):,} "
                  f"({idx/len(manifest)*100:.1f}%) | "
                  f"평균 CER: {avg_cer:.4f} | "
                  f"남은 시간: {eta/60:.1f}분")
    
    total_time = time.time() - total_start
    
    print(f"\n✅ 추론 완료!")
    print(f"⏱️  총 소요 시간: {total_time/60:.2f}분")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 결과 통계
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    valid_cers = [r['CER'] for r in results if r['CER'] is not None]
    
    if valid_cers:
        avg_cer = sum(valid_cers) / len(valid_cers)
        min_cer = min(valid_cers)
        max_cer = max(valid_cers)
        
        print(f"\n📊 검증 결과:")
        print(f"  전체 샘플:    {len(results):,}개")
        print(f"  성공:         {len(valid_cers):,}개")
        print(f"  실패:         {len(results) - len(valid_cers):,}개")
        print(f"  평균 CER:     {avg_cer:.4f}")
        print(f"  최소 CER:     {min_cer:.4f}")
        print(f"  최대 CER:     {max_cer:.4f}")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 엑셀 저장
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    # 파일명 생성: validate_yyyymmdd_hhmmss_{검증 데이터 건수}.xlsx
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"validate_{timestamp}_{len(manifest)}.xlsx"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"\n💾 엑셀 파일 저장 중...")
    
    df = pd.DataFrame(results)
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 엑셀 저장
    df.to_excel(output_path, index=False, sheet_name='Validation Results')
    
    print(f"✅ 결과 저장 완료: {output_path}")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 샘플 결과 출력 (상위 5개)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    if valid_cers:
        print(f"\n" + "=" * 70)
        print("🏆 Best 5 (CER 낮은 순):")
        print("=" * 70)
        
        df_sorted = df[df['CER'].notna()].sort_values('CER').head(5)
        for idx, row in df_sorted.iterrows():
            print(f"\n[{idx+1}] CER: {row['CER']:.4f} | 추론: {row['추론시간(초)']:.3f}초")
            print(f"  파일: {row['파일명']}")
            print(f"  정답: {row['정답']}")
            print(f"  예측: {row['예측']}")
        
        print(f"\n" + "=" * 70)
        print("⚠️  Worst 5 (CER 높은 순):")
        print("=" * 70)
        
        df_sorted = df[df['CER'].notna()].sort_values('CER', ascending=False).head(5)
        for idx, row in df_sorted.iterrows():
            print(f"\n[{idx+1}] CER: {row['CER']:.4f} | 추론: {row['추론시간(초)']:.3f}초")
            print(f"  파일: {row['파일명']}")
            print(f"  정답: {row['정답']}")
            print(f"  예측: {row['예측']}")
    
    print(f"\n✅ 모든 작업 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()