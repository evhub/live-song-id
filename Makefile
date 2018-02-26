.PHONY: setup
setup:
	pip install -U keras scipy numpy librosa tensorflow

.PHONY: install
install: setup
	pip install -e .

.PHONY: clean
clean:
	rm -rf ./dist ./build
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete

.PHONY: tensorboard
tensorboard:
	open http://localhost:6006
	tensorboard --logdir=./saved_data
