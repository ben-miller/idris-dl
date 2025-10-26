module LinearAlgebra

import Data.Vect
import Data.List

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

-- Matrix type
public export
Matrix : Nat -> Nat -> Type -> Type
Matrix m n a = Vect m (Vect n a)

-- Matrix addition
-- TODO Overload this?
export
mAdd : Num a => Matrix m n a -> Matrix m n a -> Matrix m n a
mAdd = zipWith vAdd

-- Matrix scalar multiplication
-- TODO Overload this?
export
mScale : Num a => a -> Matrix m n a -> Matrix m n a
mScale k = map (scale k)

-- Matrix transpose
export
mTranspose : { n : _ } -> Matrix m n a -> Matrix n m a
mTranspose [] = replicate n []
mTranspose (x :: xs) = let xsTrans = mTranspose xs in
                                 zipWith (::) x xsTrans
