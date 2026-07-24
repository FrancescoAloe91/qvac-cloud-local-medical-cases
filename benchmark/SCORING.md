# How ranking is calculated

Blind judge (DeepSeek R1) scores each model **per clinical section**  
(`diagnosis`, `tests`, `urgency`, `safety`, `plan`).  
The host then applies a **fixed linear formula** (the judge’s raw `score` field is ignored).

---

## Per-section score (0–96.5)

\[
\text{section} = 100 \times (0.45\cdot m + 0.20\cdot a + 0.25\cdot Q + 0.10\cdot S)
\]

then **capped at 96.5** (a literal **100% is not used**).

| Symbol | Weight | What it is | Why |
|--------|--------|------------|-----|
| **m** (must) | **45%** | Fraction of `must_include` concepts present **by clinical meaning** (synonyms OK) | Core requirements of that section — missing these hurts most |
| **a** (acceptable) | **20%** | Fraction of extra `acceptable[]` checklist points covered | Rewards completeness beyond the bare minimum |
| **Q** (quality) | **25%** | Clinical **quality of reasoning** for that section (0–1), from the judge | Checklist alone is not enough: two answers can tick the same boxes with very different diagnostic soundness |
| **S** (stem specificity) | **10%** | How many **case-specific anchors** from the stem/gold appear in the answer (host-computed) | Penalizes generic textbook paste; rewards eGFR, drug names, timelines, ECG leads, etc. |

### What **quality (Q)** means

**Quality is not “writing style”.** It is how good the **clinical judgment** is on that question:

- Is the **primary call** correct and clearly stated?
- Is the **differential / plan** coherent and prioritized?
- Does it use **case meaning** (this patient), not vague filler (“stabilize”, “get labs”, “consult”)?
- Does it avoid **dangerous** advice (`must_not` / safety traps)?

Typical strong cloud answer: **Q ≈ 0.55–0.82**.  
Near-perfect Q (>0.90) should be rare.

---

## Final accuracy (ranking %)

\[
\text{Accuracy} = \sum_i w_i \cdot \text{section}_i
\]

`w_i` = fixed **section weights** in the case JSON (example Case B):

| Section | Typical weight | Why |
|---------|----------------|-----|
| **diagnosis** | high (e.g. 35%) | Wrong primary frame fails the case |
| **safety** | high (e.g. 25%) | Discriminator (lithium–CKD / sildenafil–nitrates) |
| **tests** / **urgency** | medium | Workup and disposition |
| **plan** | lower | Initial management; still counted |

Run total is capped near **97%**. Accuracies of the four models are **always distinct**: if raw totals would tie, order is broken by  
**safety → mean quality → stem specificity → diagnosis** (documented, not random).

---

## What you see in the dashboard

1. **Ranking** = Accuracy % (formula above)  
2. **Scores by clinical dimension** = each section’s score  
3. **Why these scores** = weights, discriminators, strongest/weakest section per model  
4. Judge rationale lines show `m=… a=… quality=… spec=… → score`
