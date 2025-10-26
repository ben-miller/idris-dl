import Data.Vect
import LinearAlgebra

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
