.PHONY: watch test test_mnist watch_mnist test_nn watch_nn test_pytorch watch_pytorch test_models watch_models

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

test_pytorch:
	poetry run python test/escnn/test_pytorch_hello.py

watch_pytorch:
	watchexec -e py -- poetry run python test/escnn/test_pytorch_hello.py

test_models:
	poetry run pytest test/models/test_models.py -v

watch_models:
	watchexec -e py -- poetry run pytest test/models/test_models.py -v

