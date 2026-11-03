# Decision log

Every deviation from the approved plan or every choice that could affect results goes here
(date, decision, reason). This becomes part of the Methods / Threats-to-validity sections.

| # | Date | Decision | Reason |
|---|------|----------|--------|
| D1 | 2026-11-03 | Study area in Moscow region (AOI candidates in configs/aois.yaml), UTM 37N (EPSG:32637) | Supervisor/university location; one UTM zone covers the region |
| D2 | 2026-11-03 | Monthly composites for **April–October** (7 time steps) instead of 12 months | Moscow has snow cover ~Nov–Mar and very few clear Sentinel-2 scenes in winter; to be confirmed with scripts/02_aoi_report.py |
| D3 | 2026-11-03 | Snow (SCL 11) is kept, not masked | Snow is real surface information in April/October; masking it would empty those months |
| D4 | 2026-11-03 | Months without any valid observation are gap-filled with the training mean (0 after normalisation), identically in all systems | One simple rule that every system can apply |
| D5 | 2026-11-03 | ESA WorldCover 2021 is used as the reference label; accuracy = agreement with WorldCover, not ground truth | WorldCover is itself a model product (reported global overall accuracy about 77% for v200) → threat to validity |
| D6 | 2026-11-03 | Models are trained once, on the cloud-native cube; other systems only prepare data and run inference | Fair comparison: differences come from the systems, not from retraining |
| D7 | 2026-11-03 | In lazy systems (odc-stac, ODC) W2 cube construction and W3 masking/compositing are timed as one stage | Loading and computing happen together when the lazy graph is executed |
| D8 | 2026-11-03 | Crop-type task (Task 2) is not run for the Moscow AOI | EuroCrops does not cover Russia; no open parcel-level crop labels available for the area |
| D9 | 2026-11-03 | Test metrics use only test-block pixels at least 32 px (`split.margin_px`) away from the block edge | Same buffer as the training data; a CNN would otherwise see training pixels as context at block edges |
| D10 | 2026-11-03 | A scene that cannot be read (after 10 GDAL retries) makes that month FAIL instead of being silently skipped | Silently dropped scenes would change the composite and look like a system difference; failures are logged and the month is rerun |
| D11 | 2026-11-03 | Timing analysis uses only rows with `status == ok`; memory is reported as peak RSS sampled every 0.25 s during the stage | Interrupted/failed runs are not valid timings; end-of-stage memory would underestimate the peak |
| D12 | 2026-11-03 | GPU training is not bit-for-bit repeatable; reproducibility of models is reported as the spread over 3 seeds, and the exported ONNX files (not retraining) are what all systems use | cuDNN kernels are non-deterministic; system comparison stays exact because every system runs the same ONNX file |
