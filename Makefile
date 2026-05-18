.PHONY: all create-venv install-deps distribution augmentation transformation balance clean jupyter lint

all: create-venv install-deps

create-venv:
	python3 -m venv venv
	@printf "Virtual environment created!\n"

install-deps:
	venv/bin/pip install -r requirements.txt

distribution:
	@venv/bin/python3 scripts/distribution.py data/leaves/images

augmentation:
	@venv/bin/python3 scripts/augmentation.py "data/leaves/images/Apple_healthy/image (1).JPG"

balance:
	@venv/bin/python3 scripts/balance.py data/leaves/images --output data/augmented

transformation:
	@venv/bin/python3 scripts/transformation.py "data/leaves/images/Apple_healthy/image (1).JPG"

lint:
	@venv/bin/flake8 scripts

jupyter:
	@venv/bin/jupyter lab

clean:
	rm -rf venv results data/augmented
	find . -type d -name "__pycache__" -exec rm -rf {} +
