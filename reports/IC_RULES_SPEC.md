# IC_RULES_SPEC

## IC Domains (5)
- Sensory / Vitality / Locomotion / Cognition / Psychological

## Trigger Rules (Proxy)
**Sensory**: hearing/vision impairment (听力障碍=1 or 视力障碍=1) OR failed screening (感知-听力=0 or 感知-视力=0) OR impairment impacts daily life (听力/视力障碍是否影响日常=1).
**Vitality**: 活力-营养描述结果 <=1 OR 摄食减少<=1 OR 体重下降情况<=1 OR BMI<18.5 OR 小腿围<33.
**Locomotion**: 步态异常=1 OR 250m步行困难=1 OR 肌少症-步行困难>=1 OR 运动-总分<=9 OR 握力低 (男<28, 女<18).
**Cognition**: 认知-总分 < 24.
**Psychological**: 心理-总分 >= 5 OR 是否焦虑抑郁症 = 1.

## Missing Handling
- If a domain's contributing fields are missing, no impairment is triggered by that rule (conservative: not impaired).
- Imputation used only for modeling; IC proxy uses raw values with missing-safe logic.

## IC_total
IC_total = Σ_d IC_domain_d, d in {sensory, vitality, locomotion, cognition, psychological}.

## IC_level (Standard Risk)
- 0–1: Low risk
- 2–3: Medium risk
- 4–5: High risk
Rationale: severity increases with the count of impaired IC domains.