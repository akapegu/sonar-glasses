# 06 Phoneme analysis of the test outputs

Which sounds does the echo model get wrong? `phoneme_eval.py` converts reference and output
words to ARPABET phonemes (CMUdict, plus g2p_en's letter-to-sound model for unknown and
garbled words; 39 phonemes, no stress), aligns them per sentence and reports phoneme error
rate (PER), per-phoneme recall, the most common confusions, and a viseme error rate that
ignores confusions between sounds that look alike on the face.

```bash
.venv/bin/pip install g2p_en==2.1.0
.venv/bin/python experiments/06_phoneme_analysis/phoneme_eval.py \
    --csv <run>/logprobs_test_last_decode_final_ext_a0.75_b3.0_w128.csv --cols greedy hyp \
    --out experiments/06_phoneme_analysis/results_seed0
```

Caveat: the phonemes are derived from word (greedy: subword) outputs, not from a phoneme
recogniser, so a wrong word shows up as a run of phoneme errors that the sensor may not have
caused. Greedy outputs are the closest view of what the echo model itself reads.

## Results: final_vs seed 0, test

| Output | Mode | PER | Sub / Del / Ins | Viseme error rate | WER (Whisper) |
| --- | --- | --- | --- | --- | --- |
| greedy | voiced | 14.2% | 6.8 / 6.0 / 1.5 | 11.1% | 18.1% |
| greedy | silent | **23.7%** | 11.3 / **10.3** / 2.0 | 18.4% | 29.4% |
| external-text LM | voiced | 10.4% | 4.0 / 5.8 / 0.5 | 8.5% | 11.1% |
| external-text LM | silent | **18.4%** | 6.9 / **10.6** / 0.9 | 15.1% | 19.6% |

Phoneme recall by class (greedy):

| Class (phonemes) | Voiced | Silent | Drop |
| --- | --- | --- | --- |
| bilabial (P B M) | 90% | 83% | −7 |
| labiodental (F V) | 91% | 86% | −6 |
| approximant (R W) | 88% | 82% | −6 |
| velar (K G NG) | 86% | 77% | −9 |
| alveolar (T D N S Z L) | 88% | 77% | −10 |
| postalveolar (SH ZH CH JH) | 78% | 68% | −10 |
| dental (TH DH) | 91% | 81% | −10 |
| vowels (spread / open / round) | 85 / 86 / 88% | 75 / 79 / 80% | −10 / −8 / −7 |

Most frequent substitutions, silent greedy: D→T 77, AH→IH 57, Z→S 57, T→D 49, N→T 49, IH→AH 39,
B→M 36, T→N 33. Worst phonemes, silent: SH 61%, Z 62%, CH 64%, ER 70%, D 72%. Best: W 89%,
F 89%, AY 88%, TH 86%, M 85%, B 83%.

## What it says about the sensing

- **Lip sounds are read best, tongue sounds worst.** Bilabials, labiodentals and W lose the
  least in silent speech (6–7 points); alveolar and postalveolar sounds (T D N S Z SH CH),
  which are made with the tongue tip behind the teeth, lose the most and have the lowest
  recall (61–77%). This fits a sensor that measures surface motion (lips, cheeks, jaw) and
  barely sees the tongue.
- **Voicing and nasality are the top confusions:** D↔T, Z↔S (voicing) and N↔T, B↔M (nasality)
  share a place of articulation and differ only in the vocal folds or the soft palate, which no
  surface sensor can see. In silent speech there is no voicing at all, so these pairs can only
  be resolved by the language model.
- **At the phoneme level, deletions are as common as substitutions in silent speech** (10.3%
  vs 11.3%), and the LM does not reduce them (10.6%). The LM fixes substitutions (11.3 → 6.9)
  but not deletions. At the word level the picture is milder: deletions are about a third of
  silent errors (see the next section), and part of the phoneme-level deletion count comes from
  wrong words that are shorter than the right ones.
- **Only about a quarter of substitutions stay inside one viseme class** (25–27%; vowel classes
  here are coarse), so most errors are not simply look-alike sounds; many come from wrong or
  garbled words.

## Where word deletions happen (`deletion_analysis.py`)

```bash
.venv/bin/python experiments/06_phoneme_analysis/deletion_analysis.py \
    --csv <run>/logprobs_test_last_decode_final_ext_a0.75_b3.0_w128.csv --cols greedy hyp \
    --data data --tokenizer <run>/tokenizer.model
```

Word level, final_vs seed 0 with the external-text LM, test, silent: substitutions 11.4%,
deletions 7.2%, insertions 1.0%. We checked the premises of three proposed deletion fixes
(margins at the utterance end, 8× instead of 16× downsampling, a blank penalty) against this data.

**There is no end-of-sentence effect: the apparent one is an alignment artifact.** When an output
is one word short and the neighbouring words differ, either word can be counted as deleted at
the same cost. jiwer puts the deletion as late as possible, which piles deletions onto the last
word. Counting only deletions that land on the same word under both tie-breaks ("clean"):

| Silent, with LM | First word | Middle | Last word |
| --- | --- | --- | --- |
| jiwer-style alignment | 1.8% | 7.3% | 11.9% |
| clean (same under both tie-breaks) | 1.8% | 4.0% | 2.4% |

Two alignment-free checks agree: the reference's last word is among the output's last two
words in 76% of silent sentences (84% voiced), and of 468 silent outputs that are too short, 308
still end on the right last word and only 16 end on the second-to-last word (a truncation).
The model's output length also matches `ResNet34CTC.out_lengths` (ceil of T/16) for every input
length from 16 to 1,200 frames, so no output step is lost. Clean deletions are similar for
function and content words (3.9% vs 4.1% mid-sentence). So: no case for adding margins at the
utterance end.

**Speaking rate does not drive deletions after the LM.** Durations come from the raw
`transcript.json` files (segments are back-to-back, so a duration includes the pause before
the sentence). Silent, with LM:

| Words per second | WER | Subst. | Deletions (clean / merge) | Insertions |
| --- | --- | --- | --- | --- |
| 1.6–2.5 (slowest fifth) | 25.6% | 17.3% | 6.6% (2.9 / 3.7) | 1.7% |
| 2.5–2.8 | 21.2% | 12.7% | 7.4% (3.5 / 3.9) | 1.1% |
| 2.8–3.0 | 20.7% | 11.6% | 7.9% (4.1 / 3.8) | 1.1% |
| 3.0–3.2 | 19.1% | 10.8% | 7.5% (4.0 / 3.5) | 0.8% |
| 3.2–4.3 (fastest fifth) | 15.0% | 7.9% | 6.6% (3.9 / 2.7) | 0.5% |

The correlation of speaking rate with a sentence's deletion rate is 0.00 (voiced 0.01). At 16×
downsampling the densest test sentence needs 0.54 tokens per output step (median 0.29), well
under CTC's limit of about 1, so the model has room for every token. Tokens per step does
correlate with deletions (+0.23), but just as much with substitutions (+0.27), because it mostly
measures word rarity: rare words split into more pieces (correlation with tokens per word
+0.72). Rarity predicts deletions about 4× more strongly than speed (+2.2 vs +0.5 points per
standard deviation). In greedy outputs, before the LM, clean deletions do rise with speed
(silent 1.8% → 4.8%), but the LM recovers them. So: no case for 8× downsampling as a deletion
fix.

**What did work: a blank penalty in the decoder** (subtract a constant from the CTC blank's
log-probability before beam search; see [04_lm_decoding](../04_lm_decoding)). It cut silent
deletions from 7.2% to 4.9% (voiced 3.3% → 2.2%) at a cost of +0.5 substitutions and +0.4
insertions, and lowered test WER by 0.9 / 1.4 points (seed 0).

Token densities above use the dev smoke run's 1,000-piece tokenizer (the final run's tokenizer
is trained the same way on slightly more text), so they are approximate.
