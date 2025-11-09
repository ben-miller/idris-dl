.PHONY: watch test test_mnist watch_mnist test_nn watch_nn test_models train_mnist train_baseline train_augmented train_equivariant test_train evaluate evaluate_models

test:
	idris2 --exec main test/Test.idr

watch:
	watchexec -e idr -- idris2 -p test --exec main test/Test.idr

test_mnist:
	idris2 --exec main test/TestMNIST.idr

watch_mnist:
	watchexec -e idr -- idris2 -p test --exec main test/TestMNIST.idr

test_nn:
	idris2 --exec main test/TestNN.idr

watch_nn:
	watchexec -e idr -- idris2 -p test --exec main test/TestNN.idr

test_models:
	poetry run pytest test/models/test_models.py -v

test_train:
	poetry run pytest test/rotational_mnist/test_train.py -v -s

train_mnist:
	poetry run python scripts/rotational_mnist/train.py

train_baseline:
	poetry run python scripts/rotational_mnist/train.py baseline

train_augmented:
	poetry run python scripts/rotational_mnist/train.py augmented

train_equivariant:
	poetry run python scripts/rotational_mnist/train.py equivariant

evaluate:
	poetry run python scripts/rotational_mnist/evaluate.py

