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
- **Deletions are as common as substitutions in silent speech** (10.3% vs 11.3%), and the LM
  does not reduce them (10.6%): whole short words and unstressed syllables leave too little
  surface motion to register. The LM fixes substitutions (11.3 → 6.9) but not deletions.
- **Only about a quarter of substitutions stay inside one viseme class** (25–27%; vowel classes
  here are coarse), so most errors are not simply look-alike sounds; many come from wrong or
  garbled words.
