# spi0 환경 (2026-09-11)
- 서버: spi0.ece.utexas.edu, 8x RTX PRO 6000 Blackwell 96GB, x86_64, Ubuntu 24.04, 스케줄러 없음
- ROOT=/home/hk28965/specwm ; `source env.sh` 가 SCRATCH/HF_HOME/venv/CUDA_DEVICE_ORDER 설정
- venv: python 3.12.3, torch 2.11.0+cu128 (sm_120 확인), diffusers 0.39, transformers, torchao, einops, imageio-ffmpeg, scipy, numpy<2. flash_attn 없음 (SPECWM_NO_FLASH=1 -> SDPA)
- repo: lingbot-world (Base, v1), lingbot-world-v2 (Fast, v2) + specwm/ 커스텀 코드 81개 (TACC 그대로)
- 체크포인트: lingbot-world/lingbot-world-base-cam 150G, lingbot-world-v2/lingbot-world-v2-14b-causal-fast 81G (HF 직접, 파일수 TACC 와 동일)
- 데이터: data/ (rt2_action_*, traj55_action_*, 슬라이스, p_*.txt) -> 루트 심링크. 참조 영상 5개 specwm_out/revisit, repro
- 스크립트: scripts/ 340개. /scratch/11779/hskim99 -> $SCRATCH 치환, SBATCH 제거, .slurm -> .job.sh
- 실행: `./run.sh <gpu|auto> scripts/xxx.job.sh [tag]` -> 백그라운드, 로그 logs/. auto = 사용 <2GB 인 첫 GPU
- 스모크: Fast 161프레임 fast_alone gen_s 69.5s (TACC GH200 36s; SDPA 경로라 2배 느림). 로드 268s (첫 실행, 디스크 캐시).
  Base 4스텝 81프레임: expert 스왑 메모리 추적은 정상(35.1 GiB 올림/0.6 내림). 실제 실행은 GPU 가 전부 점유돼 미검증.
- 주의: 다른 사용자(hl35496)가 8 GPU 를 쓰는 시간대가 있음. 스케줄러가 없으니 run.sh auto 로 빈 GPU 확인 후 시작. 랩 GPU 사용 규칙은 Slack #compute 에서 조율 필요.
- 미전송: VisionReward 가중치 24G (필요시 hf download THUDM/VisionReward-Video), 산출물 66G 중 참조 외 (TACC 에 유지)
- (09-12 밤) 실험 패키지 exp/ 추가: exp/README.md 참조. VisionReward 가중치(hf/VisionReward-Video, zai-org 로 이전됨) + vr-venv + VisionReward 리포 전송 완료.
- (09-12) examples/00/image.jpg 이 upstream 에서 05(드래곤)와 동일 파일이었음. p_00.txt(만리장성)과 안 맞아 examples/04 의 만리장성 이미지로 교체 (원본은 examples/00/image_orig_dragon_dup_of_05.jpg).
