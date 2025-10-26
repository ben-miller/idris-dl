import Data.Vect
import LinearAlgebra
import Statistics
import FullyConnectedNN

-- Simple test helper
test : Eq a => Show a => String -> a -> a -> IO ()
test name expected actual =
  if expected == actual
    then putStrLn $ "✓ " ++ name
    else putStrLn $ "✗ " ++ name ++ " - expected " ++ show expected ++ " but got " ++ show actual

main : IO ()
main = do
  test "Basic addition" [5, 7, 9] (vAdd [1, 2, 3] [4, 5, 6])
  test "Vector subtraction" [3, 3, 3] (vSub [7, 8, 9] [4, 5, 6])
  test "Vector subtraction negative" (the (Vect 3 Integer) [-3, -3, -3]) (vSub [1, 2, 3] [4, 5, 6])
  test "Scalar multiplication" [2, 4, 6] (scale 2 [1, 2, 3])
  test "Scalar multiplication by zero" [0, 0, 0] (scale 0 [5, 10, 15])
  test "Scalar multiplication negative" (the (Vect 3 Integer) [-3, -6, -9]) (scale (-3) [1, 2, 3])
  test "Dot product" 32 (dot [1, 2, 3] [4, 5, 6])
  test "Matrix addition" [[5, 7], [9, 11]] (mAdd [[1, 2], [3, 4]] [[4, 5], [6, 7]])
  test "Matrix scalar multiplication" [[2, 4], [6, 8]] (mScale 2 [[1, 2], [3, 4]])
  test "Matrix scalar multiplication by zero" [[0, 0], [0, 0]] (mScale 0 [[5, 10], [15, 20]])
  test "Matrix transpose" [[1, 3], [2, 4]] (mTranspose [[1, 2], [3, 4]])
  test "Matrix-vector multiplication" [5, 11] (matVecMult [[1, 2], [3, 4]] [1, 2])
  test "Matrix multiplication" [[7, 10], [15, 22]] (matMult [[1, 2], [3, 4]] [[1, 2], [3, 4]])
  test "Mean" 3.0 (mean [1.0, 2.0, 3.0, 4.0, 5.0])

  -- Fully connected NN tests
  let nn = initTwoLayerNN 2 3 2
  test "NN output should be [0.6, 0.8]" [0.6, 0.8] (predictTwoLayer nn [1.0, 2.0])
