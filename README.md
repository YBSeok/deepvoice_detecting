# deepvoice_detecting

AI 생성 음성·음악 탐지. 제출 필드 5개: `FILE_FAKE`, `VOICE_FAKE`, `MUSIC_FAKE`, `VOICE_PRESENT`, `MUSIC_PRESENT`.

## 리더보드

최신이 위. 점수는 제출 후 기입.

| 버전 | 날짜 | 점수 | 비고 |
|------|------|------|------|
| v2 | 2026-09-18 | | PANNs VF/MF 헤드 + DF-Arena VF max |
| v1 | 2026-09-17 | | 원본 믹스 인코딩 + 마스크 풀링 |
| v0 | | | 베이스라인 (PANNs + Demucs 스템 + DF-Arena) |

## 변경 이력

### v2 — 학습된 VF/MF 헤드

추론은 원본 믹스 기준. Demucs 줄기는 DF-Arena 마스크용으로만 쓴다.

- VP/MP: 믹스 → PANNs (v1과 동일)
- MF: 믹스 → PANNs 임베딩 → `mf_head.pt`
- VF: `max(PANNs vf_head, DF-Arena)`. DF-Arena는 v1처럼 믹스 인코딩 + 보컬 마스크 풀링
- FILE_FAKE: `max(VP×VF, MP×MF)` (동일)

학습에서 v1 대비 바꾼 점:

- 가짜 악기만이 아니라 진짜 음성·진짜 노래도 MF=0으로 학습 (실음성/실곡 오탐 감소)
- `Fake_Voice_Only_TTS_ko`로 VF=1, `Voice_Only_Zeroth`로 한국어 실음성 VF=0
- 부분 fake 오버레이: 진짜음성+가짜음악, 가짜음성+진짜/가짜음악, 한국어실음성+음악
- MF는 반주 줄기가 아니라 믹스로 학습·추론을 맞춤

### v1 — 원본 인코딩 + 프레임 풀링

- VP/MP: 믹스 → PANNs (v0과 동일)
- Demucs는 재합성 스템을 탐지에 넣지 않고, 프레임 가중치만 생성
- VF/MF: 원본 믹스를 DF-Arena에 1회 인코딩한 뒤 마스크로 풀링
- FILE_FAKE: `max(VP×VF, MP×MF)` (v0과 동일)

지문을 지키기 위해 인코더 입력 파형은 수정하지 않음.

### v0 — 베이스라인

- VP/MP: 믹스 → PANNs
- VF/MF: Demucs 보컬/반주 스템을 DF-Arena에 각각 입력
- FILE_FAKE: `max(VP×VF, MP×MF)`
