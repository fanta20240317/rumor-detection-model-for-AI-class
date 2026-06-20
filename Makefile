.PHONY: install train evaluate evaluation predict test clean

PYTHON ?= python
MODEL ?= models/main_fusion.pkl
TRAIN ?= train.csv
VAL ?= val.csv
TEXT ?= Swiss museum confirms it will take on \#Gurlitt collection

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) train.py --train $(TRAIN) --val $(VAL) --model $(MODEL) --metrics outputs/metrics.json

evaluate:
	$(PYTHON) evaluate.py --model $(MODEL) --data $(VAL) --train $(TRAIN) --out-dir outputs

evaluation: evaluate

predict:
	$(PYTHON) predict.py --model $(MODEL) --train $(TRAIN) --text "$(TEXT)"

test:
	$(PYTHON) -m unittest discover -s tests

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
