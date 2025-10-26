module LinearAlgebra

import Data.Vect
import Data.List

-- Vector addition
export vAdd : Num a => Vect n a -> Vect n a -> Vect n a
vAdd = zipWith(+)

-- Vector subtraction
export vSub : Neg a => Vect n a -> Vect n a -> Vect n a
vSub = zipWith(-)

-- Vector scalar multiplication
export scale : Num a => a -> Vect n a -> Vect n a
scale k = map (* k)

-- Dot product
export dot : Num a => Vect n a -> Vect n a -> a
dot xs ys = sum $ zipWith (*) xs ys

-- MxN Matrix
public export
Matrix : Nat -> Nat -> Type -> Type
Matrix m n a = Vect m (Vect n a)

-- Matrix scalar multiplication
export mScale : Num a => a -> Matrix m n a -> Matrix m n a
mScale k = map (scale k)
