module Statistics

import Data.Vect
import LinearAlgebra

export
mean : Fractional a => { n : _ } -> Vect (S n) a -> a
mean xs = sum xs / fromInteger (cast (S n))
