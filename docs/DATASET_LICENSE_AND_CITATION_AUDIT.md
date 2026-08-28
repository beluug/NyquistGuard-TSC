# Dataset Licence and Citation Audit

Audit date: 29 August 2026  
Scope: datasets used by the V5.1 ten-dataset retrospective extension and the
four-dataset independent confirmation  
Status: **machine-assisted audit; accountable author review remains required before submission**

This record separates repository terms from the project's redistribution decision.
The project will publish code, configurations, frozen result tables and reproducibility
metadata, but will not include original datasets or processed caches containing source
signals. A `Yes` in the redistribution column describes the repository licence in
principle, not permission to bundle the data into this repository without a separate
rights review.

| Dataset | Source and authoritative URL | Access condition | Licence observed | Required citation | Redistribution / manuscript action |
|---|---|---|---|---|---|
| BasicMotions | UEA/UCR archive description: https://www.timeseriesclassification.com/description.php?Dataset=BasicMotions | Public archive URL; automated request returned HTTP 401 on 2026-08-29 | **UNRESOLVED**: licence text could not be retrieved | Cite the UEA archive paper (manuscript Ref. 20) and the archive/dataset record used | Do not redistribute; author must inspect the current archive terms and confirm citation wording. |
| Epilepsy | UEA/UCR archive description: https://www.timeseriesclassification.com/description.php?Dataset=Epilepsy | Public archive URL; automated request returned HTTP 401 on 2026-08-29 | **UNRESOLVED**: licence text could not be retrieved | Cite the UEA archive paper (manuscript Ref. 20) and the archive/dataset record used | Do not redistribute; author must inspect the current archive terms and confirm citation wording. |
| PAMAP2 | UCI dataset 231: https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring | Public UCI landing page and download | CC BY 4.0 (confirmed on official landing page) | UCI citation shown on page: Reiss (2012), DOI 10.24432/C5NW2H (manuscript Ref. 21) | Licence permits sharing/adaptation with attribution; raw data remains excluded from release. |
| MHEALTH | UCI dataset 319: https://archive.ics.uci.edu/dataset/319/mhealth+dataset | Public UCI landing page and download | CC BY 4.0 (confirmed on official landing page) | UCI citation shown on page: Banos, Garcia & Saez (2014), DOI 10.24432/C5TW22 (manuscript Ref. 22) | Licence permits sharing/adaptation with attribution; raw data remains excluded from release. |
| HAPT | UCI dataset 341: https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions | Public UCI landing page and download | CC BY 4.0 (confirmed on official landing page) | UCI citation shown on page: Reyes-Ortiz et al. (2015), DOI 10.24432/C54G7M (manuscript Ref. 23) | Licence permits sharing/adaptation with attribution; raw data remains excluded from release. |
| Daily and Sports Activities | UCI dataset 256: https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities | Public UCI landing page and download | CC BY 4.0 (confirmed on official landing page) | UCI citation shown on page: Barshan & Altun (2010), DOI 10.24432/C5C59F (manuscript Ref. 24) | Licence permits sharing/adaptation with attribution; raw data remains excluded from release. |
| Hydraulic Systems | UCI dataset 447: https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems | Public UCI landing page and download | CC BY 4.0 (confirmed on official landing page) | UCI citation shown on page: Helwig, Pignanelli & Schutze (2015), DOI 10.24432/C5CW21 (manuscript Ref. 25) | Licence permits sharing/adaptation with attribution; raw data remains excluded from release. |
| Sleep-EDF Expanded v1.0.0 | PhysioNet: https://physionet.org/content/sleep-edfx/1.0.0/ | Open access subject to the stated licence | Open Data Commons Attribution License v1.0 (confirmed on versioned page) | Dataset DOI 10.13026/C2X676 (Ref. 26) and original publication Kemp et al. (2000, Ref. 27); page also requests the current standard PhysioNet citation | Do not bundle raw files; retain source URL/DOI and add the standard PhysioNet citation if required by the final journal check. |
| EEG Motor Movement/Imagery v1.0.0 | PhysioNet: https://physionet.org/content/eegmmidb/1.0.0/ | Open access subject to the stated licence | Open Data Commons Attribution License v1.0 (confirmed on versioned page) | Dataset DOI 10.13026/C28G6P (Ref. 28) and BCI2000 publication (Ref. 29); page states both dataset and original-publication citation | Do not bundle raw files; retain source URL/DOI and confirm whether the standard PhysioNet citation must also be added. |
| MIT-BIH Arrhythmia v1.0.0 | PhysioNet: https://physionet.org/content/mitdb/1.0.0/ | Open access subject to the stated licence | Open Data Commons Attribution License v1.0 (confirmed on versioned page) | Dataset DOI 10.13026/C2F305 (Ref. 30) and Moody & Mark (2001, Ref. 31); page also requests the current standard PhysioNet citation | Do not bundle raw files; retain source URL/DOI and add the standard PhysioNet citation if required by the final journal check. |
| SelfRegulationSCP1 | Zenodo record 11206265: https://zenodo.org/records/11206265 | Public Zenodo record; official TRAIN/TEST files downloaded from record | CC BY 4.0 (`cc-by-4.0`, confirmed through Zenodo API) | Cite the record DOI 10.5281/zenodo.11206265 and the UEA archive paper (manuscript Ref. 20) | Licence permits sharing/adaptation with attribution; project will not redistribute the `.ts` files. |
| HandMovementDirection | Zenodo record 11206224: https://zenodo.org/records/11206224 | Public Zenodo record; official TRAIN/TEST files downloaded from record | CC BY 4.0 (`cc-by-4.0`, confirmed through Zenodo API) | Cite the record DOI 10.5281/zenodo.11206224 and the UEA archive paper (manuscript Ref. 20) | Licence permits sharing/adaptation with attribution; project will not redistribute the `.ts` files. |
| RacketSports | Zenodo record 11206263: https://zenodo.org/records/11206263 | Public Zenodo record; official TRAIN/TEST files downloaded from record | CC BY 4.0 (`cc-by-4.0`, confirmed through Zenodo API) | Cite the record DOI 10.5281/zenodo.11206263 and the UEA archive paper (manuscript Ref. 20) | Licence permits sharing/adaptation with attribution; project will not redistribute the `.ts` files. |
| Heartbeat | Zenodo record 11206229: https://zenodo.org/records/11206229 | Public Zenodo record; official TRAIN/TEST files downloaded from record | CC BY 4.0 (`cc-by-4.0`, confirmed through Zenodo API) | Cite the record DOI 10.5281/zenodo.11206229 and the UEA archive paper (manuscript Ref. 20) | Licence permits sharing/adaptation with attribution; project will not redistribute the `.ts` files. |

## Findings requiring author confirmation

1. BasicMotions and Epilepsy remain unresolved because the UEA description pages
   returned an access challenge during this audit. Their licence must not be inferred
   from the four Zenodo records or from the UEA archive paper.
2. The three PhysioNet pages explicitly require an original publication citation and
   display a standard PhysioNet citation. The manuscript already cites the dataset
   DOI and original paper where applicable; the author must decide, after checking the
   target journal's data policy, whether to add the current platform citation.
3. UCI and Zenodo licences allow attribution-based sharing, but the release plan still
   excludes raw data and signal-bearing processed caches to avoid unnecessary
   redistribution and to keep the repository reproducible from source links.
4. The author must record the final access date, licence URLs/versions, and any
   repository-specific acknowledgement text in the submission package.

