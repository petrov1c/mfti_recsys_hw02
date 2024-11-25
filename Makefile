.PHONY: train
train:
	PYTHONPATH=. python src/train.py config/config.yml

save:
	PYTHONPATH=. python src/save.py experiments/gpt_CE_MSE/epoch_epoch=19-val_MSE_loss=0.126.ckpt

infer:
	PYTHONPATH=. python src/infer.py

.PHONY: lint
lint:
	flake8 src/*.py

