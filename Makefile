.PHONY: all create-venv install-deps distribution augmentation balance transform train predict package signature clean jupyter lint



# Usage: make distribution src=path/to/folder
DATASET_PATH = data/leaves/images
IMAGE_PATH = data/leaves/images/Apple_Black_rot/image (1).JPG
AUGMENTED_PATH = data/augmented
TRANSFORMED_PATH = data/transformed
PACKAGE_PATH = packages/dataset.zip

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
	venv/bin/pip install -r requirements.txt

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

train:
	@PYTHONPATH="$(PYTHONPATH)" venv/bin/python3 -c "from src.classification.train import main; main()" "$(AUGMENTED_PATH)"

predict:
	@PYTHONPATH="$(PYTHONPATH)" venv/bin/python3 -c "from src.classification.predict import main; main()" "$(IMAGE_PATH)"

package:
	@mkdir -p packages
	@echo "Packaging step placeholder. Will be completed in Gate F."

signature:
	@echo "Signature step placeholder. Will be completed in Gate F."

transformation:
	@venv/bin/python3 scripts/transformation.py "data/leaves/images/Apple_healthy/image (1).JPG"

lint:
	@venv/bin/flake8 scripts

jupyter:
	@venv/bin/jupyter lab

clean:
	rm -rf venv results models packages "$(TRANSFORMED_PATH)"
	find . -type d -name "__pycache__" -exec rm -rf {} +
