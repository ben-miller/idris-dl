module Lib.MNIST

import Data.Vect
import Data.List
import System.File
import Data.Bits
import Data.Buffer

%default total

-- MNIST image dimensions
public export
ImageWidth : Nat
ImageWidth = 28

public export
ImageHeight : Nat
ImageHeight = 28

public export
ImageSize : Nat
ImageSize = ImageWidth * ImageHeight

-- Type aliases for MNIST data
public export
Image : Type
Image = Vect ImageSize Double

public export
Label : Type
Label = Fin 10

public export
Dataset : Type
Dataset = List (Image, Label)

-- Helper to read a big-endian 32-bit integer
readBigEndianInt32 : List Bits8 -> Maybe Int
readBigEndianInt32 [a, b, c, d] =
  let a' = cast {to=Int} a
      b' = cast {to=Int} b
      c' = cast {to=Int} c
      d' = cast {to=Int} d
  in Just $ (a' `shiftL` 24) .|. (b' `shiftL` 16) .|. (c' `shiftL` 8) .|. d'
readBigEndianInt32 _ = Nothing

-- Normalize pixel value from [0, 255] to [0, 1]
normalize : Bits8 -> Double
normalize b = (cast {to=Double} b) / 255.0

-- Convert list to Vect of specific length
listToVect : (n : Nat) -> List a -> Maybe (Vect n a)
listToVect Z [] = Just []
listToVect Z (_ :: _) = Nothing
listToVect (S k) [] = Nothing
listToVect (S k) (x :: xs) = map (x ::) (listToVect k xs)

-- Parse images from raw bytes
covering
parseImages : List Bits8 -> Maybe (Nat, List Image)
parseImages bytes = do
  -- Read header (16 bytes)
  let (header, rest) = splitAt 16 bytes
  magic <- readBigEndianInt32 (take 4 header)
  guard (magic == 2051)
  numImages <- readBigEndianInt32 (take 4 $ drop 4 header)
  rows <- readBigEndianInt32 (take 4 $ drop 8 header)
  cols <- readBigEndianInt32 (take 4 $ drop 12 header)
  guard (rows == cast ImageHeight && cols == cast ImageWidth)

  -- Parse images
  let parseImage : List Bits8 -> Maybe Image
      parseImage pixelBytes = do
        pixels <- listToVect ImageSize pixelBytes
        pure $ map normalize pixels

  let rec : List Bits8 -> Nat -> List Image -> Maybe (List Image)
      rec [] 0 acc = Just (reverse acc)
      rec [] _ _ = Nothing
      rec bytes (S k) acc = do
        let (imgBytes, rest) = splitAt (cast ImageSize) bytes
        img <- parseImage imgBytes
        rec rest k (img :: acc)
      rec _ _ _ = Nothing

  images <- rec rest (cast numImages) []
  pure (cast numImages, images)

-- Parse labels from raw bytes
covering
parseLabels : List Bits8 -> Maybe (Nat, List Label)
parseLabels bytes = do
  -- Read header (8 bytes)
  let (header, rest) = splitAt 8 bytes
  magic <- readBigEndianInt32 (take 4 header)
  guard (magic == 2049)
  numLabels <- readBigEndianInt32 (take 4 $ drop 4 header)

  -- Parse labels
  let parseLabel : Bits8 -> Maybe Label
      parseLabel b = natToFin (cast b) 10

  labels <- traverse parseLabel (take (cast numLabels) rest)
  pure (cast numLabels, labels)

-- Load MNIST dataset from files
export covering
loadMNIST : String -> String -> IO (Either FileError Dataset)
loadMNIST imagesPath labelsPath = do
  -- Read image file as binary
  Right imgFile <- openFile imagesPath Read
    | Left err => pure (Left err)

  Right imgSize <- fileSize imgFile
    | Left err => do
        closeFile imgFile
        pure (Left err)

  Just imgBuf <- newBuffer (cast imgSize)
    | Nothing => do
        closeFile imgFile
        pure (Left FileReadError)

  Right bytesRead <- readBufferData imgFile imgBuf 0 (cast imgSize)
    | Left err => do
        closeFile imgFile
        pure (Left err)

  closeFile imgFile

  -- Read label file as binary
  Right lblFile <- openFile labelsPath Read
    | Left err => pure (Left err)

  Right lblSize <- fileSize lblFile
    | Left err => do
        closeFile lblFile
        pure (Left err)

  Just lblBuf <- newBuffer (cast lblSize)
    | Nothing => do
        closeFile lblFile
        pure (Left FileReadError)

  Right bytesRead2 <- readBufferData lblFile lblBuf 0 (cast lblSize)
    | Left err => do
        closeFile lblFile
        pure (Left err)

  closeFile lblFile

  -- Convert buffers to byte lists
  let readBytes : Buffer -> Int -> Nat -> IO (List Bits8)
      readBytes buf offset 0 = pure []
      readBytes buf offset (S k) = do
        byte <- getBits8 buf offset
        rest <- readBytes buf (offset + 1) k
        pure (byte :: rest)

  imageBytes <- readBytes imgBuf 0 (cast imgSize)
  labelBytes <- readBytes lblBuf 0 (cast lblSize)

  -- Parse images and labels
  case (parseImages imageBytes, parseLabels labelBytes) of
    (Just (nImages, images), Just (nLabels, labels)) => do
      if nImages == nLabels
        then pure $ Right (zip images labels)
        else pure $ Left FileReadError  -- Mismatch in counts
    _ => pure $ Left FileReadError  -- Parse error

-- Helper to get a specific digit from the dataset
export
filterByDigit : Label -> Dataset -> List Image
filterByDigit digit dataset = map fst $ filter (\(_, label) => label == digit) dataset
