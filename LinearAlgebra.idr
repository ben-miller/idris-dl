module LinearAlgebra

import Data.Vect
import Data.List

-- Vector addition
export vAdd : Num a => Vect n a -> Vect n a -> Vect n a
vAdd = zipWith(+)

-- Vector subtraction
export vSub : Neg a => Vect n a -> Vect n a -> Vect n a
vSub = zipWith(-)

-- Scalar multiplication
export scale : Num a => a -> Vect n a -> Vect n a
scale k = map (* k)
