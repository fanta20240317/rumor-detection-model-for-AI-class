.PHONY: install train evaluate predict test clean

PYTHON ?= python
MODEL ?= models/ensemble.pkl
TRAIN ?= train.csv
VAL ?= val.csv
TEXT ?= Swiss museum confirms it will take on \#Gurlitt collection

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) train.py --train $(TRAIN) --val $(VAL) --model $(MODEL) --metrics outputs/metrics.json

evaluate:
	$(PYTHON) evaluate.py --model $(MODEL) --data $(VAL) --train $(TRAIN) --out-dir outputs

predict:
	$(PYTHON) predict.py --model $(MODEL) --train $(TRAIN) --text "$(TEXT)"

test:
	$(PYTHON) -m unittest discover -s tests

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
