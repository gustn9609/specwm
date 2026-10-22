# 발표자용 가이드: 10/12–13 미팅 (Guandao, Diana 교수님)

최종본 10/12 08:15 CDT. 숫자는 전부 `facts.md`(또는 거기 적힌 raw 파일) 기준이고, 노트에서만 계산한 숫자는 슬라이드에도 "derived"라고 붙입니다. 슬라이드 위 표·문구는 영어, 말할 내용은 한국어로 정리했습니다. 타이밍은 모두 RTX PRO 6000 Blackwell(96 GB) 한 장 기준이며 H100이 아닙니다.

**속도 숫자 규칙 (가장 먼저):** teacher의 예전 "4.9 fps quiet"는 clip 전체 평균(31.2 s / 13 chunk, 초반의 싼 chunk 포함)이라 steady 값이 아닙니다. steady 기준 teacher는 **quiet 4.08 fps / 부하 상태 3.9 fps**입니다. 슬라이드에서는 quiet steady 배수 **teacher-parity 1.67× / quality 2.39× / realtime 3.85×**를 말하고, bench-43 부하 상태 배수(1.74× / 2.5× / 4.1×, ~5 % 이내로 일치)는 괄호로만 씁니다. 어떤 숫자도 4.9로 나누지 마세요.

세 가지 모드 이름:
- **ours: teacher-parity mode** = chunk-wise adaptive 0.7 + safe commit (`cw07_safe`)
- **ours: quality mode** = 832, 2-step, fused + partial commit (ours 832)
- **ours: realtime mode** = 704, 2-step, fused + partial commit, 디바이스는 VAE decode + bicubic만 (ours 704)
- f02b, f03, f04, f09, f10은 이름 정하기 전에 만든 그림이라 "ours 832 / ours 704"로 표시되어 있습니다(= quality / realtime mode).

## 10/5 이후 바뀐 것 (한 문단)

10/5에는 "SDVG가 클립당 99.9초 걸리고, 저해상도는 품질이 안 나온다"는 상태였습니다. 그 뒤로 (1) SDVG를 모든 arm과 같은 최적화 스택으로 다시 돌렸지만 논문 τ에서는 teacher보다 느리고, 빨라지는 τ에서는 student 수준 품질로 떨어져서 speculative decoding(1.3B draft → 14B)은 닫았습니다. physics verifier 버전도 target-only보다 빠른 클립이 없었습니다. (2) 저해상도 손실을 분리해 보니 14B 자체가 576 아래에서 급격히 나빠지고, 704에서는 1.3B refiner가 해상도만큼 품질을 깎아서 refiner를 빼고 디바이스는 디코드만 하게 했습니다. (3) 대신 14B 하나를 직접 가속해서 세 가지 모드를 만들었습니다: teacher와 품질이 같은 1.67×, 품질 −0.30의 2.39×, 품질 −0.62의 3.85×(MG2와 같은 품질에 drift 13점 낮음). (4) fused commit이 만든 청크 경계 점프를 찾아 partial commit으로 고쳤습니다(12 % → 4 %). (5) 평가는 bench-43(86 paired clips), Chenfeng 방식 per-frame latency, StEvo-38, rollout-path 거리, 그리고 같은 조건의 quiet timing pass로 바꿨습니다. 이 과정에서 teacher 4.9 fps(정의 차이), 18.7 fps(점프 있는 설정), student ≈ 10 fps(compile 없던 옛 런) 같은 예전 숫자를 정정했습니다.

## 1. 슬라이드 순서 (본편 14장 + 백업)

| # | slide title (EN) | figure / table | on-slide message (EN, one line) |
|---|---|---|---|
| 1 | Since 10/5: what changed | `notes/s1_overview.md` "10/5 → 10/12 at a glance" table | Speculative decoding and the refiner are closed; the 14B itself is now accelerated in three operating modes. |
| 2 | Headline: three operating modes of the 14B | **`f11_modes`** + README.md §1 headline table | Teacher-level quality at 1.67× (Δ −0.07 ± 0.06); quality mode 2.39× (−0.30 ± 0.08); realtime mode 3.85× (−0.62 ± 0.09) — quiet steady, teacher 4.08 fps. |
| 3 | What runs where (teacher / SDVG / ours) | **`f01_pipelines`** | Ours keeps only the 14B in the cloud (0.763 s per chunk quiet) and the device only decodes; SDVG pays draft + verify on top of the 14B. |
| 4 | Item 1: latency in world-model metrics | **`f13_latency`** + s2 slide table (three modes vs teacher) | One GPU, decode streamed: realtime action→frame p50 1.30 s vs teacher 3.24 s (student 1.43 s); no configuration meets the 750 ms per-frame budget. |
| 5 | Item 2: speculative decoding → NO-GO | **`f07_sdvg_analysis`** + s3 physics-verifier table | At the paper τ SDVG is slower than the teacher (44.6 vs 35.8 s per clip); the physics verifier was never faster than target-only; accept-all ceiling ≈ 7.6 fps at student quality. |
| 6 | Item 3: isolating the low-res loss | **`f08_lowres_curve`** + s4 paired-steps table | 14B alone: quality 7.53 → 6.83 down to 576, then 6.17 / 5.06 at 448 / 384; at 704 the 1.3B refiner cost as much as the resolution step, so it was removed. |
| 7 | Item 4: sensitivity per lever | **`f04_speed_bridge`** + s5 §1 lever table | Speed levers multiply (1.95× · 1.22× · 1.61× = 3.85×); quality is set by resolution and commit type; chunk-wise adaptive is free. |
| 8 | New: fused commit causes chunk-boundary jumps | **`f02b_jump_frames`** + **`f02_partial_commit`** | Fused commit jumps at 12 % of boundaries (704); partial commit brings it to 4 % (separate 3 %) for ≈ +0.12–0.13 s per chunk. |
| 9 | Item 5: all arms on bench-43 | **`f05_pareto_bench43`** + s1 bench-43 headline table | Realtime mode = Matrix-Game 2.0 quality (6.90 vs 6.91) with 13 points less drift; no arm reaches ≥ 20 fps at quality ≥ 7.2. |
| 10 | Item 5: paired deltas vs the teacher | **`f06_paired_deltas`** | Only the teacher-parity mode is faster than the teacher and within noise on both axes; MG2, SDVG τ −2.2, student and Waypoint add +19 to +25 drift. |
| 11 | Item 5: StEvo-38 (benchmark protocol) | **`f12_stevo`** | Physical inaccuracy: 14B-based arms 0.55–0.71 vs student 1.00 and Waypoint 0.92; only the quality mode separates from MG2 (0.79); state evolution is at the floor for every arm. |
| 12 | Item 5: rollout paths vs the teacher | **`f10_paths_summary`** (+ `f10_paths_pouring_water_into_cup` or `f10_paths_easy_lake`) | Ours 704 is as far from the teacher as the student (1.15 vs 1.17); MG2 is clearly farther (1.45). Distance, not quality. |
| 13 | 10/10 points and the cloud/edge splits | **`f03_splits_map`** + s7 Part A table | Only splits that keep the 14B generating survive; scalability shown on MG2 for the speed levers only. |
| 14 | What is open, next steps, asks | s8 ranked plan + questions; README.md §6 open list | Main proposal: DMD fine-tune at 704; cheapest next result: uniform 3-step control; asks: raters, Gemini key, 16 vs 20 fps target, device definition. |
| B1 | Backup: why the draft cannot help | s3 §2–3 (StEvo-85 physics table, draft–target gap) | Draft tokens are 0.63 σ from the 14B (7 % within ε) vs 0.35 σ (36 %) for the 14B's own step 1; StEvo-85 physics: 1.3B 1.2 % vs 14B 32.9–48.2 %. |
| B2 | Backup: the decoder is part of the cost | `f09_decoder` | Full VAE 426 ms < 761 ms cloud chunk on a separate device GPU; one shared GPU 10.1 fps; TAEHV −0.56 / −1.06 quality. |
| B3 | Backup: the teacher measured three ways (4.9 correction) | s7 B7 table | 4.9 = whole clip, 10 prompts; 4.08 = quiet steady; 3.9 = bench-43 steady under load. |
| B4 | Backup: latency details | s2 tables (e), (e2), (f) | Same protocol for all rows; ours n = 5 (castle s1234 cold compile); 704 fused 1180 ms and 576 fused + partial 818 ms are lower but not modes. |
| B5 | Backup: direction-1 arms | `f11_modes` panel b + s5 §4 table | Head alone (cw07_sep) +0.03 ± 0.05 at 1.45×; commit policy 1.66–1.67× at −0.10 / −0.13 vs cw07_sep (borderline); cw05 −0.17 → not a mode. |
| B6 | Backup: student at 17.2 fps on the same stack | s7 B4 | The old ≈ 10 fps is superseded; the student is faster than ours, but at quality 5.83 / drift 75.1. |
| B7 | Backup: easy vs tabletop & physics | s6 slide 4 table | The student is near the teacher on easy scenes; the cloud model earns its cost on the 38 hard prompts. |
| B8 | Backup: empty text prompts on jar / castle / hourglass | s7 B5 table | Bench-43 and StEvo-38 are unaffected; all jar/castle-only results (10/5 SDVG, physics verifier, timing pass, latency profile) ran without text. |
| B9 | Backup: per-scene path figures | other `f10_paths_<scene>` | For questions on single scenes; jar and castle have n = 3. |

### 슬라이드별로 말할 것 (한 줄 + 꼭 말할 caveat)

1. 10/5의 "SDVG 99.9초, 저해상도는 안 됨"에서 출발해서, 원인을 정리했고 실제로 돌아가는 레시피가 생겼다는 흐름. 참고로 target-only는 슬라이드와 같은 방식(prepare 포함)으로 83.3 s이고, 앞서 쓴 "≈ 73 s"는 prepare 11.3 s를 뺀 값(생성 64.4 + 디코드 8.6)이라 둘 다 맞는 숫자임 — 슬라이드에서는 83.3 s(prepare 포함)로 통일.
2. 핵심 한 장. 같은 14B를 세 모드로 돌림: 1.67× 품질 동급 / 2.39× 품질 −0.30 / 3.85× 품질 −0.62. caveat: teacher-parity는 "동급(tie)"이지 "더 좋다"가 아님(두 Δ 모두 95 % CI가 0 포함). quality/realtime 배수는 cloud GPU 시간만 센 것이라 teacher와 완전히 같은 기준이 아님(같은 GPU 직렬이면 2.25× / 3.44×, bench-43 부하). 품질 근거인 VLM quality prompt는 사람 검증이 없음.
3. 클라우드는 14B만, 디바이스는 디코드 + bicubic만. caveat: 디바이스 쪽(426 ms full VAE)이 cloud chunk 안에 숨는다는 건 제안이고 end-to-end로 잰 적 없음. 실제 엣지 기기·네트워크는 측정에 없음.
4. 10/5 피드백대로 TTFF / steady fps / action→frame / 10초 클립 시간을 Chenfeng 방식으로 측정. realtime p50 1.30 s, teacher 3.24 s. caveat: 750 ms는 아무 설정도 못 맞춤(realtime miss 96.7 %). teacher-parity는 streaming decode가 없어 per-frame 행이 없음. ours는 704에서 디코드해서 디코드가 싸다는 이점이 있음. TTFF(≈ 2.57 s)는 가정한 2.0 s prepare를 더한 derived 값.
5. Speculative decoding은 no-go. caveat: 10/5에 받은 테스트 그대로("physics verifier SDVG의 최종 영상이 physics에서 이기는가")는 결과 영상이 10/9에 지워져서 측정하지 못했다고 먼저 솔직히 말하기. 재실행은 follow-up으로 제안. τ −2.2는 bench-43에서 teacher보다 빠르지만(steady 6.7 vs 3.9 fps) 품질 −0.84로 student 쪽.
6. 14B만 native로 돌린 곡선. 576 아래 급락, 384에서 drift 급증 → 10/5의 384 실패는 저해상도 14B 자체 문제. caveat: 공식 지원 해상도는 480×832, 720×1280뿐이라 704 이하는 전부 지원 범위 밖. 448/384는 quiet fps 없음.
7. 속도 레버는 곱해지고 품질은 해상도와 commit 방식이 정함. chunk-wise adaptive는 공짜(12×3 +0.00 ± 0.11, bench-43 +0.03 ± 0.05). caveat: 점프 없는 commit(fused + partial)도 704에서 −0.36으로 해상도 한 단계(−0.31)만큼 비쌈. uniform 3-step 대조군이 없어서 learned head 자체의 기여는 증명 안 됨.
8. fused commit 때문에 경계 점프(12 %)가 생겼고 partial commit으로 4 %. 비용은 quiet에서 +0.13 s/chunk(같은 5 clip끼리는 ≈ +0.12). caveat: 예전 18.7 fps는 점프가 있는 설정이라 철회. VLM judge는 8프레임만 봐서 점프를 못 봄 → 그래서 점프 지표를 따로 만듦.
9. MG2와 품질은 같고(paired −0.01 ± 0.10) drift는 13점 낮음(paired −13.1 ± 3.6). caveat: MG2가 더 빠름(20.0 vs 15.8, quiet 19.7 vs 15.73). MG2는 torch.compile 없이 돌렸고(공식 H100 25 fps), 게임 모델이라 텍스트를 못 받고 카메라 경로를 키보드/마우스로 근사함. fps 측정 방식이 arm마다 다름(MG2/Waypoint는 자체 디코더 포함 whole-clip).
10. teacher보다 빠르면서 두 축 모두 noise 안인 건 teacher-parity mode뿐. caveat: SDVG τ −0.7도 noise 안이지만 teacher보다 느림. bench-43은 9/43 프롬프트가 개발 세트와 겹침(나머지 34개만 봐도 순서 같음, derived).
11. StEvo-38: 의미 있는 칸은 physical inaccuracy뿐. caveat: 로컬 Qwen3-VL-32B judge·seed 1개·38 task라 논문 Gemini 리더보드와 비교 불가, ~0.15 미만 차이는 유의하지 않음. state evolution / task success는 모든 arm이 바닥. "teacher보다 물리적으로 낫다"(quality 0.55 vs 0.68, p = 0.23)나 "realtime이 MG2보다 낫다"(p = 0.42)는 말하지 말 것. teacher-parity mode는 StEvo에서 안 돌렸음.
12. 경로 거리: ours 704 ≈ student, MG2가 확실히 멂. caveat: 품질이 아니라 teacher와의 거리. ours 832는 teacher와 같은 noise를 써서 비교 대상 아님. 1보다 작다고 더 좋은 게 아님(Waypoint jar 0.97, 품질 4.03). jar/castle은 n = 3.
13. 10/10의 split 세 개 중 둘은 닫고 하나(클라우드 저해상도 + 디바이스 디코드)만 남음. caveat: scalability는 MG2에서 speed lever와 drift head만 확인, commit lever·chunk-wise head는 아직 안 옮김, 차세대 base model은 안 돌림. Waypoint의 같은 해상도 비교와 student 704는 없음.
14. 다음 단계와 결정 요청. ≥ 20 fps · 품질 ≥ 7.2는 학습(DMD at 704) 없이는 안 됨. 가장 싼 다음 결과는 uniform 3-step control. 교수님들께: 평가자(Diana), Gemini API key, 목표 16 vs 20 fps, 디바이스 정의, parity를 safe(1.67×)로 부를지 separate(1.45×, Δ +0.03 ± 0.05)로 부를지, 논문 framing. 576 fused + partial은 25.66 fps quiet로 예산을 맞추지만 품질 점수가 없다는 것도 언급.

## 2. 꼭 소리 내서 말해야 할 caveat

1. **속도 배수는 quiet steady(1.67× / 2.39× / 3.85×)로 통일.** teacher 4.9는 whole-clip 평균이라 쓰지 않음. "부하 때문에 4.9 → 3.9로 떨어졌다"는 틀린 말(steady로는 4.08 vs 3.9, 3 % 차이).
2. **quality/realtime의 fps는 cloud GPU만 센 값.** 같은 GPU에서 디바이스 단계를 직렬로 세면 8.9 / 13.6 fps, full-VAE 디코드까지 같은 GPU에서 스트리밍하면 output 6.9 / 10.8 fps. teacher-parity만 teacher와 같은 기준.
3. **realtime도 실시간에 살짝 못 미침:** 0.763 s/chunk vs 재생 0.75 s. per-frame 750 ms 예산은 아무 설정도 못 맞춤.
4. **fps 기준이 arm마다 다름:** teacher/student/ours는 steady(디코드 제외), SDVG는 whole-clip(chunk 0 포함, steady로는 3.1 / 6.7), MG2/Waypoint는 자체 디코더 포함 whole-clip. 같은 기준 비교는 클립당 생성 시간(SDVG 44.6 vs teacher 35.8 s).
5. **VLM quality prompt는 사람 검증이 없음**(drift prompt만 AUROC 0.82). teacher-parity 주장이 여기에 기대고 있고, 블라인드 스터디는 아직 평가자가 없음.
6. **Physics verifier의 go 기준은 측정 못 함**(영상 10/9 삭제). no-go 근거는 비용과 proxy(StEvo, draft–target gap).
7. **uniform 3-step 대조군 없음:** learned head가 고정 3 step보다 나은지는 미증명.
8. **student(17.2 fps)가 모든 모드보다 빠름.** 메시지는 "student급 속도에서 훨씬 높은 품질"(6.90 vs 5.83, drift 56.9 vs 75.1).
9. **jar / castle / hourglass는 텍스트 프롬프트가 빈 상태로 돌았음.** bench-43과 StEvo-38은 영향 없음. quiet timing pass, latency profile, 10/5 SDVG, physics verifier, jar/castle 경로 그림은 텍스트 없이 돈 결과.
10. **quiet timing은 jar + sandcastle 2개 프롬프트**(ours quality/realtime은 castle s1234가 cold compile이라 제외, n = 5). 품질 Δ는 항상 bench-43.
11. **704 이하는 공식 지원 해상도 밖**, 실제 엣지 기기와 네트워크는 어떤 측정에도 없음.
12. **StEvo-38은 로컬 judge·seed 1개.** 리더보드와 비교 불가, state evolution은 모든 arm이 바닥.

## 3. 아직 열려 있는 것 (질문 나오면 이렇게)

- teacher-parity의 per-frame latency: spec_early에 streaming decode가 없어서 없음.
- uniform 3-step 대조군, DMD at 704, 1.3B distillation("draft를 더 세게"), idle-action trick: 계획/아이디어 단계.
- quiet timing이 없는 것: student, SDVG, 448/384(diag run cloud 시간만, 부하 기록 없음), MG2/Waypoint(자체 파이프라인 whole-clip만).
- StEvo: teacher-parity와 SDVG τ −2.2 미실행, Gemini judge는 API key 필요.
- 576 fused + partial: 25.66 fps quiet지만 품질 점수·점프 수 없음.
- Waypoint 같은 해상도 비교, student 704, MG2에 commit lever 이식: 없음.
- 자세한 목록은 README.md §6.

## 4. 이번 점검에서 고친 것 (요약)

- 노트의 MG2 quiet fps 19.5 → 19.7(facts §15 정정 반영; 19.5는 warm-up run 포함 값).
- partial commit 비용: 0.630 vs 0.763은 6 vs 5 clip median이라는 점과 같은 clip 기준 ≈ 0.12 s를 s5, s7에 추가.
- s7 B2 "704 손실은 대부분 refiner 탓" → "refiner와 해상도가 비슷한 크기(차이 유의하지 않음)".
- s8의 f11 설명(07:44 버전) 갱신, Waypoint StEvo 문구를 f12와 맞춤. s7 MG2 token 절감 29 % → 30 %.
- facts.md §1, §2, §10, §12의 오래된 PENDING 문구를 §13–§15로 연결(숫자 변경 없음).
- 그림 재생성(데이터 변화 없음): f01, f03, f05, f07 footer의 bench-43 설명을 "38 StEvo camera-WM tasks + 5 easy scenes"로, f12 제목은 MG2 대비로는 quality mode만 구분된다고 수정.
