# deepvoice_detecting

AI 생성 음성·음악 탐지. 제출 필드 5개: `FILE_FAKE`, `VOICE_FAKE`, `MUSIC_FAKE`, `VOICE_PRESENT`, `MUSIC_PRESENT`.

## 리더보드

최신이 위. 점수는 제출 후 기입.

| 버전 | 날짜 | 점수 | 비고 |
|------|------|------|------|
| v1 | 2026-09-17 | | 원본 믹스 인코딩 + 마스크 풀링 |
| v0 | | | 베이스라인 (PANNs + Demucs 스템 + DF-Arena) |

## 변경 이력

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
