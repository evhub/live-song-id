.PHONY: setup
setup:
	-cd ..; git clone https://github.com/evhub/song-annotation-db; cd live-song-id
	pip install -e ../song-annotation-db
	pip install keras scipy numpy librosa tensorflow kapre

.PHONY: clean
clean:
	rm -rf ./dist ./build
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete

.PHONY: tensorboard
tensorboard:
	open http://localhost:6006
	tensorboard --logdir=./saved_data
