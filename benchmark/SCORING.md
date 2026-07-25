# How ranking is calculated

Blind judge (DeepSeek R1) scores each model **per clinical section**  
(`diagnosis`, `tests`, `urgency`, `safety`, `plan`).  
The host then applies a **fixed linear formula** (the judge’s raw `score` field is ignored).

The judge grades **clinical meaning** (same thesis, same intent, synonyms / near-equivalents).  
It does **not** atomize the gold into keyword checklists or demand exact acronyms.

## Ground truth: gold vs rubric

| Situation | Ground truth | Effect |
|-----------|--------------|--------|
| **Confirmed diagnosis / gold pasted** (Case C always; A/B optional) | **GOLD wins** | Judge returns continuous `alignment` (semantic closeness to the gold thesis for that section) + `quality`. Empty teaching rubrics are ignored. |
| **No gold pasted** (typical A/B) | **Rubric wins** | Soft checklist on `must_include[]` / `acceptable[]` by meaning + `quality`. |

Near-perfect alignment can land in the **80–95%** band; a literal **100%** is still unused (caps below).

---

## Gold mode — per-section score (0–96.5)

\[
\text{section} = 100 \times (0.50\cdot A + 0.30\cdot Q + 0.20\cdot S)
\]

then **capped at 96.5**.

| Symbol | Weight | What it is |
|--------|--------|------------|
| **A** (alignment) | **50%** | Holistic semantic closeness to the gold for that section: diagnosis framing, workup intent, advice, next steps. Near-equivalent formulations count. |
| **Q** (quality) | **30%** | Clinical judgment quality (correct primary call, coherent plan, case-specific, not dangerous). |
| **S** (stem specificity) | **20%** | Host-computed: case anchors from stem/gold present in the answer (anti-generic-paste). |

### What the judge must evaluate (gold)

1. **Diagnosis / framing** — same clinical thesis?  
2. **Tests / workup** — same investigative intent?  
3. **Urgency** — same acuity band / red-flag thinking?  
4. **Safety / advice** — same traps avoided?  
5. **Plan / next steps** — same stepwise strategy?

Partial credit when the idea is right but incomplete. Wrong primary frame (e.g. fibromyalgia-as-primary) still hurts hard.

---

## Rubric mode — per-section score (0–96.5)

\[
\text{section} = 100 \times (0.30\cdot m + 0.20\cdot a + 0.40\cdot Q + 0.10\cdot S)
\]

| Symbol | Weight | What it is |
|--------|--------|------------|
| **m** (must) | **30%** | Fraction of rubric must concepts by **meaning** |
| **a** (acceptable) | **20%** | Extra completeness |
| **Q** (quality) | **40%** | Clinical judgment |
| **S** (stem specificity) | **10%** | Case-specific anchors |

Quality dominates over checklist so near-synonyms are not punished.

---

## Final accuracy (ranking %)

\[
\text{Accuracy} = \sum_i w_i \cdot \text{section}_i
\]

`w_i` = fixed **section weights** in the case JSON. Run total capped near **97%**.  
Ties broken by **safety → mean quality → stem specificity → diagnosis**.

---

## What you see in the dashboard

1. **Ranking** = Accuracy %  
2. **Scores by clinical dimension** = each section’s score  
3. **Why these scores** = weights / strongest–weakest sections  
4. Rationale lines show `align=… quality=… spec=… → score` (gold) or `m=… a=… quality=… → score` (rubric)
