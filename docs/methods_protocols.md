# Splitting protocols: methods text (R4 draft)

**Status:** draft for the proposal and thesis methodology chapter. All numbers were produced by our scripts on the sanitised HCRL table (`src/validity.py`, `src/run_technique_b.py`, Hafiz's audit scripts) and cross-checked against the temporal audit. Items marked **[PENDING]** wait for a supervisor decision and must not be presented as final.

---

## 1. Data, labels and signatures

We use the HCRL Car-Hacking dataset: one attack-free (ambient) recording and four attack recordings (DoS, Fuzzy, RPM spoofing, Gear spoofing). Each frame has a timestamp, a CAN identifier, a data length code (DLC), up to eight data bytes and, in the attack files, a flag R (regular) or T (injected). A frame is labelled with the attack class of its file if its flag is T and Normal otherwise; every frame of the ambient file is Normal. The five classes are always reported in the order Normal, DoS, Fuzzy, RPM, Gear.

The attack files end in a long attack-free stretch that repeats the same recording in all four files. These tails (891,068 frames per file, 3,564,272 in total) are removed before any split, because a chronological test set would otherwise contain no attack frames for DoS, Fuzzy and Gear. The sanitised table has 13,994,074 frames: Normal 11,662,557, DoS 587,521, Fuzzy 491,847, RPM 654,897, Gear 597,252. **[PENDING]** The sanitisation rule (cut at the largest timestamp gap above 1.0 s) is provisional until the supervisor confirms it.

The **signature** of a frame is the tuple (identifier, DLC, data bytes 0 to 7); the timestamp is not part of it, and it is never a model input. The sanitised table contains 975,878 distinct signatures and no signature occurs under two labels. The classes differ sharply in their redundancy: DoS, RPM and Gear are each a single repeated message (one signature per class), Fuzzy frames are all different (one signature per frame), and Normal is heavily repeated but diverse (484,028 signatures).

A test frame is **seen** if its exact signature occurs in the training set and **unseen** otherwise. Seen frames can be classified by memorisation; unseen frames test generalisation.

## 2. Why one universal signature-group split is not used

A group split places all frames with the same signature on one side. This cannot be applied to all five classes: DoS, RPM and Gear consist of a single signature, so the whole class would land either in training or in testing and the split would be invalid for three of the five classes. For Fuzzy every frame is its own signature, so a group split is identical to a random split. It differs from a random split only for Normal. On the supervisor's advice we therefore compare two techniques that can be applied to every class and analyse memorisation and temporal leakage separately.

## 3. Technique A: random split (baseline)

A shuffled random 80/20 split of the frames. For each class c, exactly round(0.2 · N_c) frames are drawn at random without replacement into the test set and all other frames form the training set, so each class keeps the same test share. A seed fixes the draw, and the seeds 42, 43, 44, 45 and 46 are used. On the sanitised table every seed gives 11,195,261 training and 2,798,813 test frames (test share 0.2000). Technique A is the "flawed baseline": frames recorded within milliseconds of each other, and many exact copies, can fall on both sides of the split.

Under Technique A the share of Normal test frames that are seen is 0.97101 (mean over five seeds, SD 0.00008), equal to the value 0.97098 predicted by the formula L = Σ m (1 − 0.2^(m−1)) / N_c, where m is the number of copies of a signature. DoS, RPM and Gear are always 100% seen and Fuzzy is always 0% seen.

## 4. Technique B: chronological block split (robust proposal)

Timestamps are comparable only inside one recording, so each file is split separately and training data always occur entirely before test data. For a file with first timestamp t0 and last timestamp t_end (after sanitisation):

1. Attack bursts are found in the T frames: a gap of more than 1.0 s between consecutive T frames starts a new burst. Each attack file contains exactly 300 bursts.
2. The 80% point τ = t0 + 0.8 (t_end − t0) is computed.
3. If τ falls inside a burst it is moved to the quiet stretch after that burst; otherwise the quiet stretch containing τ is used. The cut is placed at the midpoint of that quiet stretch, so no burst is cut.
4. Frames with t ≤ cut are training frames and frames with t > cut are test frames. No frames are discarded around the cut.

The ambient file has no T frames and is cut at 80% of its duration (405.149 s). The cuts for the attack files are at 1864.503 s (DoS), 1972.794 s (Fuzzy), 1566.145 s (RPM) and 1562.469 s (Gear) after the start of the file, each in a quiet stretch of about 3 s. For DoS, RPM and Gear the 80% point lay inside a burst and was moved; for Fuzzy it lay in a quiet stretch.

The result is 11,210,909 training and 2,783,165 test frames (test share 0.1989). The test shares per class are Normal 0.2011, DoS 0.1849, Fuzzy 0.1548, RPM 0.2063 and Gear 0.1984. The share of Normal test frames that are seen is 0.9145 under Technique B, against 0.9710 under Technique A. The split is deterministic: seeds change only the randomness of the models. **[PENDING]** Whether a single cut with five model seeds is sufficient, or whether several cut points (rolling origin) are needed to report variance for Technique B, is an open supervisor question.

## 5. Validity rule

A split is valid if every class occurs on both sides and each class's test share lies within a tolerance, provisionally 10% to 30%. An invalid split is reported as a limitation and is never forced. All six splits (Technique A with five seeds, Technique B) are valid under this rule; the smallest class share is Fuzzy under Technique B (0.1548). **[PENDING]** The tolerance must be confirmed by the supervisor and fixed before any model is trained on the real data.

## 6. Separating memorisation from temporal leakage

For class c let M_c be the share of its test frames that are seen, and let R_c(seen) and R_c(unseen) be the recall of class c on seen and on unseen test frames.

- Memorisation effect (within one split) = R_c(seen) − R_c(unseen).
- Temporal effect = R_c(unseen | Technique A) − R_c(unseen | Technique B).

A lookup baseline predicts the training label of the exact signature and Normal for an unseen signature; its score measures pure memorisation, and a model that beats it on unseen frames has generalised. Because DoS, RPM and Gear have one signature, they have no unseen test frames and these effects are undefined for them; likewise Fuzzy has no seen frames. The effects are therefore informative mainly for Normal (and, for generalisation, for Fuzzy). The unseen frames of the two techniques are different subpopulations, so the comparison is descriptive.

## 7. Common evaluation rules

Both techniques use the same features (identifier, DLC and the eight data bytes as ten integers), the same models and the same fixed hyperparameters, and the same metrics. Models: Logistic Regression, Decision Tree, Random Forest and the lookup baseline. **[PENDING]** The model set and the hyperparameters are placeholders until the supervisor confirms the model families and the settings are frozen. Metrics: macro-F1 over the fixed five classes (F1 = 2TP / (2TP + FP + FN)), per-class recall (undefined, not zero, for a class without test frames), accuracy always next to the all-Normal baseline 2π_N / (C (1 + π_N)), and recall on seen and unseen frames. The baseline is computed from each test set: 0.1818 for Technique A and about 0.183 for Technique B. Statistics are descriptive: mean and sample standard deviation over the five seeds, without significance tests. Technique B's standard deviation reflects model randomness only.

## 8. Limitations of the splits

- Technique B uses a single cut per file, so its result depends on that cut and has no split variance.
- DoS, RPM and Gear are single messages and are 100% seen under every split, so no split can separate memorisation from generalisation for them.
- The regular (R) frames of the attack files and of the ambient file come from different recording conditions, so a model could learn the recording condition instead of the attack.
- The validity tolerance, the sanitisation rule and the model set are provisional until the supervisor decides.
- Seen and unseen test frames are different subpopulations, so the memorisation and temporal effects are descriptive and not causal.
