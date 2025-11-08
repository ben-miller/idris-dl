module Main

import Lib.MNIST
import Data.Vect
import Data.String
import System.File

-- Display a simple ASCII representation of an image
showImage : Image -> String
showImage img =
  let rows = reshape {outer=28} {inner=28} img
      showPixel : Double -> Char
      showPixel p = if p > 0.5 then '#' else ' '
      showRow : Vect 28 Double -> String
      showRow row = pack (toList (map showPixel row))
  in unlines (toList (map showRow rows))
  where
    reshape : {outer, inner : Nat} -> Vect (outer * inner) a -> Vect outer (Vect inner a)
    reshape {outer=Z} [] = []
    reshape {outer=S k} {inner} xs =
      let (row, rest) = splitAt inner xs
      in row :: reshape rest

main : IO ()
main = do
  putStrLn "Loading MNIST data..."

  -- Load training data
  result <- loadMNIST "data/train-images-idx3-ubyte"
                      "data/train-labels-idx1-ubyte"

  case result of
    Left err => putStrLn $ "Error loading data: " ++ show err
    Right dataset => do
      putStrLn $ "Successfully loaded " ++ show (length dataset) ++ " training samples"

      -- Show first few examples
      let samples = take 3 dataset
      for_ (zip [1..3] samples) $ \(idx, (img, label)) => do
        putStrLn $ "\nSample " ++ show idx ++ ": Label = " ++ show (finToNat label)
        putStrLn $ showImage img

      -- Load test data
      testResult <- loadMNIST "data/t10k-images-idx3-ubyte"
                              "data/t10k-labels-idx1-ubyte"

      case testResult of
        Left err => putStrLn "Error loading test data"
        Right testDataset => do
          putStrLn $ "\nSuccessfully loaded " ++ show (length testDataset) ++ " test samples"
          putStrLn "MNIST data is ready for your neural network!"
