CONFIG_FILE=config/config.yaml

.PHONY: default train

default:
	@echo "Pretraining Bart"

train:
	@echo "Working on..."

compress:
	cd ..; tar -czvf pre-training-bart{.tar.gz,}
