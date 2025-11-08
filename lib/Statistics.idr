module Lib.Statistics

import Data.Vect
import Lib.LinearAlgebra

export
mean : Fractional a => { n : _ } -> Vect (S n) a -> a
mean xs = sum xs / fromInteger (cast (S n))
