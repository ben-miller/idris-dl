module LinearAlgebra

import Data.Vect

-- Vector addition
-- TODO Overload this?
export
vAdd : Num a => Vect n a -> Vect n a -> Vect n a
vAdd = zipWith (+)

-- Vector subtraction
-- TODO Overload this?
export
vSub : Neg a => Vect n a -> Vect n a -> Vect n a
vSub = zipWith (-)

-- Vector scalar multiplication
-- TODO Overload this?
export
scale : Num a => a -> Vect n a -> Vect n a
scale k = map (* k)

-- Dot product
export
dot : Num a => Vect n a -> Vect n a -> a
dot xs ys = sum $ zipWith (*) xs ys

-- Element-wise multiplication (Hadamard product)
export
vMul : Num a => Vect n a -> Vect n a -> Vect n a
vMul = zipWith (*)

-- Matrix type
public export
Matrix : Nat -> Nat -> Type -> Type
Matrix m n a = Vect m (Vect n a)

-- Matrix addition
-- TODO Overload this?
export
mAdd : Num a => Matrix m n a -> Matrix m n a -> Matrix m n a
mAdd = zipWith vAdd

-- Matrix subtraction
-- TODO Overload this?
export
mSub : Neg a => Matrix m n a -> Matrix m n a -> Matrix m n a
mSub = zipWith vSub

-- Matrix scalar multiplication
-- TODO Overload this?
export
mScale : Num a => a -> Matrix m n a -> Matrix m n a
mScale k = map (scale k)

-- Matrix transpose
export
mTranspose : { n : _ } -> Matrix m n a -> Matrix n m a
mTranspose [] = replicate n []
mTranspose (x :: xs) = let xsTrans = mTranspose xs in zipWith (::) x xsTrans

-- Matrix-vector multiplication
export
matVecMult : Num a => Matrix m n a -> Vect n a -> Vect m a
matVecMult mat v = map (dot v) mat

-- Matrix multiplication
export
matMult : Num a => { p : _ } -> Matrix m n a -> Matrix n p a -> Matrix m p a
matMult mat1 mat2 =
  let mat2T = mTranspose mat2
  in map (\row => map (dot row) mat2T) mat1

-- Outer product: create a matrix from two vectors
export
outerProduct : Num a => Vect m a -> Vect n a -> Matrix m n a
outerProduct v1 v2 = map (\x => map (* x) v2) v1
