# D2 IC Proxy Rules (v0.1)

## Overview
This IC proxy uses 5 domains (sensory, vitality, locomotion, cognition, psychological). Each domain is flagged as impaired (1) if any proxy rule triggers. IC_total is the sum of impaired domains (0–5).

## Domain Rules
- Sensory: hearing/vision impairment (听力障碍/视力障碍 = 1) OR failed screening (感知-听力/感知-视力 = 0) OR impairment impacts daily life (听力/视力障碍是否影响日常 = 1).
- Vitality: malnutrition or risk (活力-营养描述结果 <= 1) OR reduced intake (活力-过去三个月内...摄食减少 <= 1) OR weight loss (活力-过去三个月内体重下降情况 <= 1) OR BMI < 18.5 OR calf circumference < 33 cm.
- Locomotion: gait abnormal (步态异常-编码 = 1) OR cannot walk 250m (衰弱快速筛查量表-1 = 1) OR sarcopenia walking difficulty (肌少症评估-2 >= 1) OR SPPB total (运动-总分 <= 9) OR low grip strength (male < 28, female < 18).
- Cognition: cognitive total score (认知-总分 < 24).
- Psychological: GDS-15 total (心理-总分 >= 5) OR diagnosis (是否焦虑抑郁症 = 1).

## IC Level Mapping
- IC_level: standard risk mapping by impairment count (0–1 Low, 2–3 Medium, 4–5 High).
- IC_level_inv: inverse mapping (0–1 High, 2–3 Medium, 4–5 Low) retained for stakeholder comparison.