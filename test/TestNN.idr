module Main

import Lib.MNIST
import Lib.FullyConnectedNN
import Data.Vect
import Data.String
import System.File
import Lib.LinearAlgebra

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

-- Train network on a dataset for one epoch
trainEpoch : Double -> TwoLayerNN ImageSize 128 10 -> List (Image, Label) -> TwoLayerNN ImageSize 128 10
trainEpoch lr nn [] = nn
trainEpoch lr nn ((img, lbl) :: rest) =
  let target = labelToOneHot lbl
      updatedNN = trainStep lr nn (img, target)
  in trainEpoch lr updatedNN rest

-- Train for multiple epochs
trainEpochs : Nat -> Double -> TwoLayerNN ImageSize 128 10 -> List (Image, Label) -> IO (TwoLayerNN ImageSize 128 10)
trainEpochs 0 lr nn dataset = pure nn
trainEpochs (S k) lr nn dataset = do
  putStrLn $ "Epoch " ++ show (S k) ++ "..."
  let trainedNN = trainEpoch lr nn dataset
  trainEpochs k lr trainedNN dataset

main : IO ()
main = do
  putStrLn "Loading MNIST training data..."

  -- Load training data
  trainResult <- loadMNIST "data/train-images-idx3-ubyte"
                           "data/train-labels-idx1-ubyte"

  case trainResult of
    Left err => putStrLn $ "Error loading training data: " ++ show err
    Right fullTrainDataset => do
      putStrLn $ "Loaded " ++ show (length fullTrainDataset) ++ " training samples"

      -- Load test data
      putStrLn "Loading MNIST test data..."
      testResult <- loadMNIST "data/t10k-images-idx3-ubyte"
                              "data/t10k-labels-idx1-ubyte"

      case testResult of
        Left err => putStrLn $ "Error loading test data: " ++ show err
        Right testDataset => do
          putStrLn $ "Loaded " ++ show (length testDataset) ++ " test samples\n"

          -- Use subset for faster training
          let trainDataset = take 1000 fullTrainDataset
          let testSamples = take 100 testDataset

          -- Initialize network with Xavier initialization
          nn <- initTwoLayerNN ImageSize 128 10

          putStrLn "Testing untrained network..."
          let initialAcc = accuracy nn testSamples
          putStrLn $ "Initial accuracy: " ++ show (initialAcc * 100.0) ++ "%\n"

          -- Train the network
          putStrLn "Training network on 1000 samples..."
          let learningRate = 0.5
          let numEpochs = 3

          trainedNN <- trainEpochs numEpochs learningRate nn trainDataset

          -- Evaluate trained network
          putStrLn "\nEvaluating trained network..."
          let finalAcc = accuracy trainedNN testSamples
          putStrLn $ "Final accuracy: " ++ show (finalAcc * 100.0) ++ "%"
          putStrLn $ "Improvement: " ++ show ((finalAcc - initialAcc) * 100.0) ++ "%\n"

          -- Show some predictions
          putStrLn "Sample predictions:"
          for_ (zip [1..5] (take 5 testSamples)) $ \(idx, (img, actual)) => do
            let pred = argmax (predictTwoLayer trainedNN img)
            let status = if pred == actual then "✓" else "✗"
            putStrLn $ "  " ++ status ++ " Sample " ++ show idx
                       ++ ": Predicted " ++ show (finToNat pred)
                       ++ ", Actual " ++ show (finToNat actual)
