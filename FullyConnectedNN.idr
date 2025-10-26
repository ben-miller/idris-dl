module FullyConnectedNN

import Data.Vect
import LinearAlgebra

-- Activation functions
export
sigmoid : Double -> Double
sigmoid x = 1.0 / (1.0 + exp (-x))

export
relu : Double -> Double
relu x = if x > 0.0 then x else 0.0

export
vSigmoid : Vect n Double -> Vect n Double
vSigmoid = map sigmoid

export
vRelu : Vect n Double -> Vect n Double
vRelu = map relu

-- A simple fully connected layer
public export
record FCLayer (input : Nat) (output : Nat) where
  constructor MkFCLayer
  weights : Matrix output input Double
  biases : Vect output Double

-- Forward pass through a single layer
export
forward : FCLayer input output -> Vect input Double -> Vect output Double
forward layer input =
  let weighted = matVecMult layer.weights input
  in vAdd weighted layer.biases

-- A two-layer fully connected network
public export
record TwoLayerNN (input : Nat) (hidden : Nat) (output : Nat) where
  constructor MkTwoLayerNN
  layer1 : FCLayer input hidden
  layer2 : FCLayer hidden output

-- Forward pass through the two-layer network
export
predictTwoLayer : TwoLayerNN input hidden output -> Vect input Double -> Vect output Double
predictTwoLayer nn input =
  let hidden = vSigmoid (forward nn.layer1 input)
      output = vSigmoid (forward nn.layer2 hidden)
  in output

-- Initialize a layer with zeros (dummy initialization)
export
initLayer : (input : Nat) -> (output : Nat) -> FCLayer input output
initLayer input output = MkFCLayer (replicate output (replicate input 0.0)) (replicate output 0.0)

-- Initialize a two-layer network with zeros
export
initTwoLayerNN : (input : Nat) -> (hidden : Nat) -> (output : Nat) -> TwoLayerNN input hidden output
initTwoLayerNN input hidden output =
  MkTwoLayerNN (initLayer input hidden) (initLayer hidden output)
