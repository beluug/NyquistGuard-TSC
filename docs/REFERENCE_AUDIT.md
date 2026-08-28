# Reference Audit

Audit date: 28 August 2026  
Scope: machine-assisted metadata review for the ESWA-targeted manuscript  
Status: **human verification still required before submission**

## Changes made

The unresolved direct sampling-rate literature placeholder was replaced with the following records after DOI/metadata lookup:

1. Hasegawa T. Smartphone sensor-based human activity recognition robust to different sampling rates. *IEEE Sensors Journal*. 2021;21(5):6930-6941. https://doi.org/10.1109/JSEN.2020.3038281
2. Khan A, Hammerla N, Mellor S, Plötz T. Optimising sampling rates for accelerometer-based human activity recognition. *Pattern Recognition Letters*. 2016;73:33-40. https://doi.org/10.1016/j.patrec.2016.01.001
3. Cheng W, Erfani S, Zhang R, Kotagiri R. Learning datum-wise sampling frequency for energy-efficient human activity recognition. *Proceedings of the AAAI Conference on Artificial Intelligence*. 2018;32(1). https://doi.org/10.1609/aaai.v32i1.11862
4. Wang J, Zhu T, Gan J, Chen LL, Ning H, Wan Y. Sensor data augmentation by resampling in contrastive learning for human activity recognition. *IEEE Sensors Journal*. 2022;22(23):22994-23008. https://doi.org/10.1109/JSEN.2022.3214198

The BCI2000 record was completed with DOI https://doi.org/10.1109/TBME.2004.827072.

## Claim-level use

- Khan et al. are cited for task-specific sampling-rate optimization.
- Cheng et al. are cited for datum-wise rate selection under an accuracy--energy objective.
- Hasegawa is cited for downsampling augmentation and adversarial rate confusion in cross-rate human activity recognition.
- Wang et al. are cited for resampling augmentation in contrastive human activity recognition.
- The manuscript distinguishes those primarily HAR-focused approaches from NyquistGuard's known-rate, physical-time filtering and analytic observability across multiple sensing domains.

## Machine-check boundary

- Crossref/DOI metadata were available for the principal journal and conference references checked during this pass.
- UCI `10.24432/...` and PhysioNet `10.13026/...` dataset identifiers use repository registration infrastructure and should be verified through their official landing pages or DataCite, not inferred invalid from a Crossref miss.
- arXiv, PMLR, OpenReview, NeurIPS, and repository records must be checked at their authoritative landing pages.
- The Holm JSTOR DOI and all dataset access/licence terms remain part of final human verification.

## Machine-assisted verification (2026-08-29)

The following checks used authoritative landing pages or DOI metadata endpoints. They
are an audit aid, not accountable-author sign-off. `PASS` means that the identifier,
title/venue/year and the principal bibliographic fields agreed with the manuscript;
the author must still open the source and confirm the exact citation and claim mapping.

| Ref. | Status | Evidence checked | Note |
|---:|:---:|---|---|
| 1 | PASS | Crossref DOI | Title, authors, *Data Mining and Knowledge Discovery* 33(4), 917-963, 2019. |
| 2 | PASS | arXiv:1803.01271 | Official record confirms Bai, Kolter, Koltun and 2018 title. |
| 3 | PASS | Crossref DOI | IJCNN 2017, pp. 1578-1585. |
| 4 | PASS | Crossref DOI | *Data Mining and Knowledge Discovery* 34(5), 1454-1495. |
| 5 | PASS | Crossref DOI | SIGKDD 2021, pp. 248-257. |
| 6 | PASS | Crossref DOI | *Data Mining and Knowledge Discovery* 36(5), 1623-1646. |
| 7 | PASS | Crossref DOI | *Data Mining and Knowledge Discovery* 35(2), 401-449. |
| 8 | PASS | arXiv:1910.04341 | Official record confirms Tan, Petitjean, Keogh, Webb and 2019 title. |
| 9 | PASS | Crossref DOI | *Proceedings of the IRE* 37(1), 10-21. |
| 10 | PASS | PMLR v97 landing page | Official proceedings page and 2019 metadata reachable. |
| 11 | PASS | arXiv:2008.02397 | Official record confirms title and four listed authors. |
| 12 | PASS | Crossref DOI | *IEEE Sensors Journal* 21(5), 6930-6941. |
| 13 | PASS | Crossref DOI | SLT 2018, pp. 1021-1028. |
| 14 | PASS | OpenReview landing page | Official forum page reachable; final author/venue fields still require manual check. |
| 15 | PASS | PMLR v70 landing page | Official proceedings page and 2017 metadata reachable. |
| 16 | PASS | NeurIPS 2017 landing page | Official proceedings page reachable; final page/article metadata requires manual check. |
| 17 | PASS | PMLR v97 landing page | Official proceedings page, 2019, pp. 2151-2159. |
| 18 | UNRESOLVED | Crossref/OpenAlex DOI metadata | Both machine records expose first page 80 only, while the manuscript says 80-83; author must inspect the JSTOR scan and correct only with evidence. |
| 19 | UNRESOLVED | JSTOR landing page and DOI | DOI API lookup returned 404 and the JSTOR page presented a client challenge; volume/issue/pages require manual verification. |
| 20 | PASS | arXiv:1811.00075 | Official record confirms 2018 archive paper and listed lead authors. |
| 21 | PASS | UCI dataset 231 landing page | Official citation and CC BY 4.0 licence confirmed. |
| 22 | PASS | UCI dataset 319 landing page | Official citation and CC BY 4.0 licence confirmed. |
| 23 | PASS | UCI dataset 341 landing page | Official citation and CC BY 4.0 licence confirmed. |
| 24 | PASS | UCI dataset 256 landing page | Official citation and CC BY 4.0 licence confirmed. |
| 25 | PASS | UCI dataset 447 landing page | Official citation and CC BY 4.0 licence confirmed. |
| 26 | PASS | PhysioNet Sleep-EDF v1.0.0 page | ODC-By 1.0, open access policy, dataset DOI and required original-paper citation confirmed. |
| 27 | PASS | Crossref DOI | *IEEE TBME* 47(9), 1185-1194, 2000. |
| 28 | PASS | PhysioNet EEGMMI v1.0.0 page | ODC-By 1.0, dataset citation and BCI2000 citation requirement confirmed. |
| 29 | PASS | Crossref DOI | *IEEE TBME* 51(6), 1034-1043, 2004. |
| 30 | PASS | PhysioNet MIT-BIH v1.0.0 page | ODC-By 1.0, open access policy, dataset DOI and required original-paper citation confirmed. |
| 31 | PASS | Crossref DOI | *IEEE EMBS Magazine* 20(3), 45-50, 2001. |
| 32 | PASS | Crossref DOI | *Pattern Recognition Letters* 73, 33-40, 2016. |
| 33 | PASS | Crossref DOI | AAAI-18 volume 32(1), 2018; article-number pagination should be checked in the official page. |
| 34 | PASS | Crossref DOI | *IEEE Sensors Journal* 22(23), 22994-23008. |

UCI pages were queried directly at their dataset landing URLs. PhysioNet pages were
queried directly at the versioned project URLs. The DOI API does not treat UCI and
PhysioNet repository identifiers as ordinary Crossref works, so their official pages,
not Crossref 404 responses, are the controlling evidence.

## Required human sign-off

Before submission, an accountable author must open every reference and confirm author order, accents/transliteration, title, venue, year, volume/issue, pagination or article number, DOI/URL, and that the cited source supports the adjacent manuscript statement. This file records an audit aid, not completed scholarly approval.
