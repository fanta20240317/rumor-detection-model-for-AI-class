# Windows Command Notes

Makefile exposes make train, make evaluate, make predict, make web, and make
test as the primary reproducibility commands.

On Windows PowerShell, use python directly if make is unavailable. The Makefile remains the reference command map.

PowerShell examples:

```powershell
python train.py --train train.csv --val val.csv --model models/main_fusion.pkl --metrics outputs/metrics.json
python evaluate.py --model models/main_fusion.pkl --data val.csv --train train.csv --out-dir outputs
python predict.py --model models/main_fusion.pkl --train train.csv --text "sample text"
python web_app.py --model models/main_fusion.pkl --train train.csv
```

School LLM environment variables can be set before running predict or web:

```powershell
$env:SCHOOL_LLM_API_KEY="your-school-api-key"
$env:SCHOOL_LLM_BASE_URL="https://school-api.example/v1"
$env:SCHOOL_LLM_MODEL="school-model-name"
```
