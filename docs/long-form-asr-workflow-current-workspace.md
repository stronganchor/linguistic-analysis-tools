# Zazaki ASR Fine-Tuning Workflow

This short guide explains the full loop for improving your fine‑tuned Whisper model.

---

## 📁 Folder structure
```
audio_corpus/
├─ audio_wav16k/        # 16kHz mono WAVs
├─ asr_output/          # Auto transcripts (.txt / .srt / .tsv)
├─ corrected_segments/  # TSV exports from ELAN
├─ training_data/
│   ├─ clips/           # 15–20s clips from corrected segments
│   └─ manifest.tsv     # Audio → Text mapping for training
├─ transcribe_with_ft.py
├─ process_latest_segments.py
├─ train_from_manifest.py
└─ venv311/
```

External model storage:

```text
%Z_MODELS_DIR%\audio_corpus
├─ zazaki_ft_001/
├─ zazaki_ft_002/
└─ zazaki_ft_003/   # (latest)
```

The audio-corpus training/transcription scripts now default to that external folder.
Set `Z_AUDIO_CORPUS_MODELS_DIR` if you want a different location.

---

## ⚙️ 0. Activate environment
```bat
cd %Z_WORKSPACE%\audio_corpus
venv311\Scripts\activate
```

---

## 🎧 1. Transcribe audio
```bat
python transcribe_with_ft.py
```
- Paste the WAV path when asked (must be in `audio_wav16k/`)
- The script chunks ~15–20s on silence and saves:
  ```
  asr_output\Name.txt
  asr_output\Name.srt
  asr_output\Name.tsv
  ```

---

## ✏️ 2. Correct in ELAN
1. File → Import → Tab‑delimited text → pick the `.tsv`
2. Check:
   - **Specify delimiter** → Tab
   - **Specify first row of data** → 2
   - Column mapping:
     - Tier → 1  
     - Begin Time → 2  
     - End Time → 3  
     - Annotation → 4
3. Add a new tier `transcription_clean`
4. Copy + correct each annotation there
5. Export that tier only (File > Export as > Tab delimited text > select correction tier > select begin/end time columns) →  
   `corrected_segments\Name_corrected.tsv`

---

## 🪚 3. Build training clips
```bat
python train_whisper.py
```
This will:
- Detect your newest corrected TSV
- Pair it with the matching WAV
- Cut segments into `training_data\clips\...wav`
- Append new rows to `training_data\manifest.tsv`

---

## 🧠 4. Fine‑tune Whisper again
Open `train_from_manifest.py` and set:
```python
BASE_MODEL_DIR = r"%Z_MODELS_DIR%\audio_corpus\zazaki_ft_002"
OUTPUT_DIR = r"%Z_MODELS_DIR%\audio_corpus\zazaki_ft_003"
```
Then run:
```bat
python train_from_manifest.py
```
→ Produces a new fine‑tuned model folder.

---

## 🚀 5. Use the new model
In `transcribe_with_ft.py`, update:
```python
MODEL_DIR = r"%Z_MODELS_DIR%\audio_corpus\zazaki_ft_003"
```
Then transcribe new audio with improved accuracy.

---

## 🔁 Summary loop
1. Transcribe new audio  
2. Correct in ELAN (`transcription_clean`)  
3. Export corrected TSV  
4. Run `process_latest_segments.py`  
5. Run `train_from_manifest.py`  
6. Update `MODEL_DIR` → repeat

Each cycle expands your dataset and improves recognition of your dialect + spelling conventions.
