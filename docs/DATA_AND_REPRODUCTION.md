# Data and Reproduction

This release contains no original datasets and no processed caches containing source
signals. Download data from the authoritative repositories, accept their terms, and
recreate local representations with the code in `src/nyquistguard/data/`.

## Sources and terms

- BasicMotions and Epilepsy: UEA/UCR archive descriptions at
  `https://www.timeseriesclassification.com/description.php?Dataset=BasicMotions` and
  `https://www.timeseriesclassification.com/description.php?Dataset=Epilepsy`.
  The archive licence must be checked at download time; the release does not
  redistribute these files.
- PAMAP2, MHEALTH, HAPT, Daily and Sports Activities, and Condition Monitoring of
  Hydraulic Systems: UCI dataset landing pages linked in
  `DATASET_LICENSE_AND_CITATION_AUDIT.md` in this directory. The UCI pages currently
  state CC BY 4.0 and provide the required dataset citations.
- Sleep-EDF Expanded, EEG Motor Movement/Imagery, and MIT-BIH Arrhythmia: versioned
  PhysioNet pages linked in `DATASET_LICENSE_AND_CITATION_AUDIT.md`. The pages
  currently state Open Data Commons Attribution License v1.0 and require dataset and
  original-publication citations.
- SelfRegulationSCP1, HandMovementDirection, RacketSports, and Heartbeat: Zenodo
  records 11206265, 11206224, 11206263 and 11206229. Each record currently reports
  CC BY 4.0; cite the record DOI and the UEA archive paper.

## Reproduction boundary

The frozen reports are descriptive evidence bound to the recorded protocols and source
hashes. Unit tests and report inspection are safe without data. Full training or test
evaluation requires the original datasets, a compatible environment and explicit
manual confirmation for guarded stages. Test data must never be used to select a
checkpoint, threshold or reliability mode.
