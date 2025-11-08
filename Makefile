.PHONY: watch test test_mnist watch_mnist test_nn watch_nn

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

