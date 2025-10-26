module Main

import MNIST
import FullyConnectedNN
import Data.Vect
import Data.String
import System.File
import LinearAlgebra

-- Convert a label (Fin 10) to a one-hot vector
labelToOneHot : Label -> Vect 10 Double
labelToOneHot lbl =
  let idx = finToNat lbl
  in Data.Vect.Fin.tabulate (\i => if finToNat i == idx then 1.0 else 0.0)

-- Get the predicted class (argmax)
argmax : {n : Nat} -> Vect (S n) Double -> Fin (S n)
argmax xs =
  let indexed = zip range xs
      maxPair = foldl1 (\(i1, v1), (i2, v2) => if v2 > v1 then (i2, v2) else (i1, v1)) indexed
  in fst maxPair

-- Compute accuracy on a dataset
accuracy : TwoLayerNN ImageSize 128 10 -> List (Image, Label) -> Double
accuracy nn dataset =
  let predictions = map (\(img, lbl) => (argmax (predictTwoLayer nn img), lbl)) dataset
      correct = length $ filter (\(pred, actual) => pred == actual) predictions
      numTotal = length dataset
  in (cast {to=Double} correct) / (cast {to=Double} numTotal)

main : IO ()
main = do
  putStrLn "Loading MNIST test data..."

  -- Load test data
  result <- loadMNIST "python/t10k-images-idx3-ubyte"
                      "python/t10k-labels-idx1-ubyte"

  case result of
    Left err => putStrLn $ "Error loading data: " ++ show err
    Right testDataset => do
      putStrLn $ "Loaded " ++ show (length testDataset) ++ " test samples"

      -- Initialize a dummy network (all zeros)
      let nn = initTwoLayerNN ImageSize 128 10

      putStrLn "\nTesting network with zero initialization..."

      -- Test on first 100 samples
      let testSamples = take 100 testDataset
      let acc = accuracy nn testSamples

      putStrLn $ "Accuracy on 100 test samples: " ++ show (acc * 100.0) ++ "%"
      putStrLn $ "Expected: ~10% (random guessing for 10 classes)"
      putStrLn $ "\nTest FAILED: Network needs proper training!"

      -- Show some predictions
      putStrLn "\nFirst 5 predictions:"
      for_ (zip [1..5] (take 5 testSamples)) $ \(idx, (img, actual)) => do
        let pred = argmax (predictTwoLayer nn img)
        putStrLn $ "  Sample " ++ show idx ++ ": Predicted " ++ show (finToNat pred)
                   ++ ", Actual " ++ show (finToNat actual)
