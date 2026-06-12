12 June 2026

Dr Stephen Johnson
Editor-in-Chief
PeerJ

Dear Dr Johnson,

I am pleased to submit my manuscript, **"EEG–Hemodynamic Coupling During General Anaesthesia: A Large-Scale Exploratory Analysis in 2,070 Surgical Cases with Null Acute Event-Response Findings,"** for consideration as an Original Research article in *PeerJ*.

During general anaesthesia, the brain and cardiovascular system change in a coupled manner, yet whether their dynamic coupling varies systematically across patients and responds to acute haemodynamic events has not been characterised at scale. Using the VitalDB open dataset, I analysed 2,070 surgical cases with synchronised raw EEG (128 Hz) and invasive arterial blood pressure (500 Hz), quantifying coupling with Granger causality and cross-wavelet coherence, and testing whether a within-patient window-level coupling score changes at surrogate haemodynamic excursions.

The central result is **predominantly null**: a per-event coupling change was not robust to the choice of correlation structure (GEE-Exchangeable *p* = 0.047 vs GEE-AR(1) *p* = 0.230; Cohen's *d* = 0.047, negligible), and all stricter event definitions yielded null or negative effects. I believe this work is well suited to *PeerJ* for three reasons:

1. **A rigorously reported negative result with methodological value.** I show that the commonly used MAP/BIS-based "event" definition captures nearly all cases (98.1%) and is therefore non-specific; the endpoint fails before any coupling metric can be fairly tested. This is an actionable lesson for the design of future intraoperative coupling and nociception-monitoring studies, and it is reported transparently rather than over-interpreting a borderline finding. *PeerJ*'s policy of evaluating soundness rather than perceived impact, and its explicit openness to negative and null results, make it the natural home for this study.

2. **Full reproducibility and open data.** The analysis is built entirely on the publicly available VitalDB dataset, and the complete Python pipeline (signal processing, Granger causality, cross-wavelet coherence, score construction, and statistics) is openly released on GitHub, consistent with *PeerJ*'s open-science requirements.

3. **Methodological transparency.** I defined the objectives before event-response modeling and provide sensitivity analyses for the event definition, the GEE correlation structure, the Granger-causality lag selection, wavelet edge effects, and the EEG feature pair, and clearly distinguish the descriptive case-level composite score from the within-patient window-level score used for event analysis.

This manuscript is original, has not been published elsewhere, and is not under consideration by any other journal. As the sole author, I have approved the submission and declare no competing interests. The work is a secondary analysis of fully de-identified, publicly available data; ethics details are provided in the manuscript.

I thank you for considering this submission and look forward to your response.

Sincerely,

Ge Gao
Attending Physician
Department of Anesthesiology
The First Affiliated Hospital, Zhejiang University School of Medicine
Hangzhou, China
gaoge2018@zju.edu.cn
