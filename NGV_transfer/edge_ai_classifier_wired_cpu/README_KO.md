# NGV CPU 실습 R6 — Windows↔WSL은 탐색기, Pi 전송은 scp

전체 순서는 NGV_CPU_Simple_Guide_20261007_R6.html을 브라우저로 여세요.

2026-10-08 Pi 설치 수정: OS의 무관한 패키지까지 검사하던 전역 pip check를 제거했습니다.
실습용 LiteRT/OpenCV/NumPy/Pillow 로딩·버전 검사는 유지합니다.
학습·변환·추론 코드와 기존 R6 가이드 순서는 그대로 사용합니다.
R6는 전송 가이드·배포 문서를 변경했고, 코드 갱신 도구의 배포 문서 이름만 R6로 맞췄습니다.

## 강사 준비 (Windows 마우스로)
1. 코드 ZIP을 바탕화면에서 모두 압축 풉니다.
2. 프로젝트 offline/raw에 가이드의 공식 원본7개를 붙여넣습니다. gz는 풀지 않습니다.
3. edge_ai_classifier_wired_cpu 폴더 자체를 ZIP으로 압축하고 NGV_CPU_CLASS_R6.zip으로 배포합니다.
Python이나 패키지 설치 파일은 USB에 준비하지 않습니다.

## 강사·학생 공통
- A=Windows PowerShell, B=Pi SSH, C=기존 WSL Ubuntu를 직접 엽니다.
- 바탕화면 ZIP을 풀고 탐색기로 WSL 홈에 프로젝트 폴더를 복사합니다.
- 바탕화면 NGV_transfer 폴더는 Windows와 Pi 사이의 파일 전송용입니다.
- Windows↔WSL은 마우스 복사, Windows↔Pi는 A 창의 scp 명령입니다.
- Pi에는 설치 전 깨끗한 프로젝트 폴더를 scp로 복사합니다.
- C: cd ~/edge_ai_classifier_wired_cpu 다음 bash setup_laptop.sh
- B: cd ~/edge_ai_classifier_wired_cpu 다음 bash setup_pi.sh
- Python·TensorFlow·ONNX·Pi 패키지는 각자 인터넷으로 설치합니다.
- 라벨 지정: C에서 source day3/env_day2.sh 다음 python tools/configure_course.py
- course_config.json을 Windows NGV_transfer로 마우스 복사한 뒤 scp로 Pi에 보냅니다.

## Ubuntu와 Python
Ubuntu24.04를 요구하지 않습니다. 실제 Ubuntu·CPU·glibc를 검사하며,
현재 Ubuntu apt 저장소의 기본 라이브러리와 프로젝트 전용 Python3.12.15를 사용합니다.
시스템 Python과 OS는 교체하지 않습니다. 노트북 조건은 Ubuntu/x86_64/AVX/
glibc2.27 이상/GLIBCXX3.4.22 이상입니다. 모든 OS와 CPU를 무조건 지원하지 않습니다.
Pi는 본인 OS의 apt와 pip로 설치하며64비트aarch64/Python3.11~3.13을 검사합니다.

## 실습
L1 카메라 실제 영상 확인
L2 사용자 라벨 촬영→CPU학습→FP32/dynamic/fullINT8→Pi→시각화
L3 ImageNet 사전학습 모델→같은 프레임의 사용자 모델 비교·분석
L4 합성 진동 파형/FFT→MLP가지치기→TFLite9개/ONNX6개→Pi→분석
L5 Fashion-MNIST CNN/가지치기/대조군→TFLite6개/ONNX4개→Pi→분석
L6 마우스로 결과 보관

환경 전환: source day3/env_day2.sh 또는 source day3/env_tools.sh
촬영 수 검사: python tools/check_capture.py
카메라 모델3정밀도 val/test평가: python wsl/evaluate_all.py
CPU 전용이며 TensorRT는 사용하지 않습니다.
설치한 가상환경을 다른 장비에 복사하지 말고 지정한 데이터·모델·결과만 전송합니다.
검증 범위는 VALIDATION_CPU.txt에 있습니다.
