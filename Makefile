.PHONY: all create-venv install-deps setup distribution augmentation balance transform transform-batch train predict package signature clean jupyter lint verify



# Usage: make distribution src=path/to/folder
DATASET_PATH = data/leaves/images
IMAGE_PATH = data/leaves/images/Apple_Black_rot/image (1).JPG
AUGMENTED_PATH = data/augmented
TRANSFORMED_PATH = data/transformed
PACKAGE_PATH = packages/dataset.zip
SIGNATURE_PATH = signature.txt

PYTHONPATH = .

ifneq ($(src),)
DATASET_PATH := $(src)
endif

ifneq ($(img),)
IMAGE_PATH := $(img)
endif

all: predict

create-venv:
	python3 -m venv venv
	@printf "Virtual environment created!\n"

install-deps:
	@test -x venv/bin/python3 || (echo "Error: missing venv. Run 'make create-venv' or 'make setup' first." && exit 1)
	venv/bin/python3 -m pip install --upgrade pip
	venv/bin/python3 -m pip install -r requirements.txt

setup: create-venv install-deps

distribution:
	@PYTHONPATH="$(PYTHONPATH)" venv/bin/python3 -c "from src.data_analysis.distribution import analyze_dataset; import sys; analyze_dataset(sys.argv[1])" "$(DATASET_PATH)"

%:
	@:

augmentation:
	@PYTHONPATH="$(PYTHONPATH)" venv/bin/python3 -c "from src.data_analysis.main import run; import sys; run(sys.argv[1], '$(AUGMENTED_PATH)')" "$(IMAGE_PATH)"

balance:
	@PYTHONPATH="$(PYTHONPATH)" venv/bin/python3 -c "from src.data_analysis.main import run; import sys; run(sys.argv[1], sys.argv[2])" "$(DATASET_PATH)" "$(AUGMENTED_PATH)"

transform:
	@PYTHONPATH="$(PYTHONPATH)" venv/bin/python3 -c "from src.data_transformation.transformation import main; main()" "$(IMAGE_PATH)"

transform-batch:
	@PYTHONPATH="$(PYTHONPATH)" venv/bin/python3 -c "from src.data_transformation.transformation import main; main()" -src "$(DATASET_PATH)" -dst "$(TRANSFORMED_PATH)" -mask

train:
	@PYTHONPATH="$(PYTHONPATH)" venv/bin/python3 -u -c "from src.classification.train import main; main()" "$(AUGMENTED_PATH)"

predict:
	@PYTHONPATH="$(PYTHONPATH)" venv/bin/python3 -c "from src.classification.predict import main; main()" "$(IMAGE_PATH)"

package:
	@mkdir -p packages
	@test -d "$(AUGMENTED_PATH)" || (echo "Error: missing $(AUGMENTED_PATH). Run augmentation/balance first." && exit 1)
	@test -f "models/leaf_model.joblib" || (echo "Error: missing models/leaf_model.joblib. Run training first." && exit 1)
	@rm -f "$(PACKAGE_PATH)"
	@{ \
		find "$(AUGMENTED_PATH)" -type f; \
		if [ -d "$(TRANSFORMED_PATH)" ]; then find "$(TRANSFORMED_PATH)" -type f; fi; \
		printf '%s\n' "models/leaf_model.joblib" "results/classification/validation_report.json" "results/classification/validation_report.txt"; \
	} | sort | zip -X -q "$(PACKAGE_PATH)" -@
	@echo "Package created: $(PACKAGE_PATH)"

signature:
	@test -f "$(PACKAGE_PATH)" || (echo "Error: missing $(PACKAGE_PATH). Run 'make package' first." && exit 1)
	@sha1sum "$(PACKAGE_PATH)" > "$(SIGNATURE_PATH)"
	@echo "Signature saved: $(SIGNATURE_PATH)"

verify:
	@sha1sum -c "$(SIGNATURE_PATH)"

lint:
	@venv/bin/flake8 scripts

jupyter:
	@venv/bin/jupyter lab

clean:
	rm -rf data/augmented data/transformed results/classification models/leaf_model.joblib packages/dataset.zip signature.txt
	rm -rf venv results models packages "$(TRANSFORMED_PATH)"
	find . -type d -name "__pycache__" -exec rm -rf {} +
