# Phase 1.5 FP/free-frame correction

The 1.5 report conflated two different denominators under the same name
"FP/free-frame". Corrected statement:

- **Frame-fired (fired_matrix), conf 0.05 (best-F1):** `fp_per_logo_free_frame
  0.192` = 10 of 52 logo-free frames produced ≥1 detection. Counts a free frame
  once if anything fired. This is a *frame* rate.
- **Box-level (sweep `fp_per_nonlogo_image`, and the 1.6 paired table):**
  conf 0.05 → **0.385**; conf 0.10 → **0.192**. Counts every detection on
  logo-free frames, divided by 52 free frames. This is a *box* rate.

Same name, different numbers, because one counts frames and the other counts
boxes. The 1.5 report's "0.192" was the frame-fired number, not the paired
table's 0.192-at-0.10 box number — the prose picked one reading without saying
which.

Rule going forward (frozen, used by `owlv2_1p7_grade.py`): **FP/free-frame is
the box-level rate** (`fp` detections on logo-free frames ÷ `n_free`). The
frame-fired rate is reported separately as "free frames with ≥1 detection",
never merged.

Source numbers: `owlv2_separation_val.json` `fired_matrix` (frame-fired) and
`sweep[].fp_per_nonlogo_image` (box-level); both on weights/logo_detector val,
36 logo / 52 logo-free frames.