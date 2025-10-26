module FullyConnectedNN

import Data.Vect
import LinearAlgebra
import Statistics

-- Activation functions
export
sigmoid : Double -> Double
sigmoid x = 1.0 / (1.0 + exp (-x))

-- Derivative of sigmoid: σ'(x) = σ(x) * (1 - σ(x))
export
sigmoidDerivative : Double -> Double
sigmoidDerivative x =
  let s = sigmoid x
  in s * (1.0 - s)

export
relu : Double -> Double
relu x = if x > 0.0 then x else 0.0

export
vSigmoid : Vect n Double -> Vect n Double
vSigmoid = map sigmoid

export
vSigmoidDerivative : Vect n Double -> Vect n Double
vSigmoidDerivative = map sigmoidDerivative

export
vRelu : Vect n Double -> Vect n Double
vRelu = map relu

-- Loss functions
-- Mean Squared Error loss
export
mseLoss : {n : Nat} -> Vect (S n) Double -> Vect (S n) Double -> Double
mseLoss predicted target =
  let diff = vSub predicted target
      squaredErrors = vMul diff diff
  in mean squaredErrors

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

-- Store intermediate values during forward pass for backpropagation
public export
record ForwardCache (input : Nat) (hidden : Nat) (output : Nat) where
  constructor MkCache
  inputVec : Vect input Double
  hiddenPreActivation : Vect hidden Double
  hiddenActivation : Vect hidden Double
  outputPreActivation : Vect output Double
  outputActivation : Vect output Double

-- Forward pass through the two-layer network
export
predictTwoLayer : TwoLayerNN input hidden output -> Vect input Double -> Vect output Double
predictTwoLayer nn input =
  let hidden = vSigmoid (forward nn.layer1 input)
      output = vSigmoid (forward nn.layer2 hidden)
  in output

-- Forward pass that saves intermediate values for backpropagation
export
forwardWithCache : TwoLayerNN input hidden output
                -> Vect input Double
                -> ForwardCache input hidden output
forwardWithCache nn input =
  let hiddenPre = forward nn.layer1 input
      hiddenAct = vSigmoid hiddenPre
      outputPre = forward nn.layer2 hiddenAct
      outputAct = vSigmoid outputPre
  in MkCache input hiddenPre hiddenAct outputPre outputAct

-- Gradients for all weights and biases
public export
record Gradients (input : Nat) (hidden : Nat) (output : Nat) where
  constructor MkGradients
  layer1WeightGrad : Matrix hidden input Double
  layer1BiasGrad : Vect hidden Double
  layer2WeightGrad : Matrix output hidden Double
  layer2BiasGrad : Vect output Double

-- Compute gradients via backpropagation (MSE loss)
export
backward : {inp, hid, out : Nat}
        -> TwoLayerNN inp hid out
        -> ForwardCache inp hid out
        -> Vect out Double  -- target
        -> Gradients inp hid out
backward nn cache target =
  -- Output layer gradients
  -- dL/dOutput = 2 * (predicted - target) / n
  let outputError = vSub cache.outputActivation target
      -- For MSE: gradient of loss w.r.t. output activation
      -- Multiply by sigmoid derivative: dL/dOutputPre = dL/dOutput * sigmoid'(outputPre)
      outputDelta = vMul outputError (vSigmoidDerivative cache.outputPreActivation)

      -- Layer 2 weight gradients: outer product of delta and hidden activation
      layer2WeightGrad = outerProduct outputDelta cache.hiddenActivation
      -- Layer 2 bias gradients: just the delta
      layer2BiasGrad = outputDelta

      -- Hidden layer gradients
      -- Backpropagate error through layer 2 weights
      layer2WeightsT = mTranspose (nn.layer2.weights)
      hiddenError = matVecMult layer2WeightsT outputDelta
      -- Multiply by sigmoid derivative
      hiddenDelta = vMul hiddenError (vSigmoidDerivative cache.hiddenPreActivation)

      -- Layer 1 weight gradients: outer product of delta and input
      layer1WeightGrad = outerProduct hiddenDelta cache.inputVec
      -- Layer 1 bias gradients: just the delta
      layer1BiasGrad = hiddenDelta

  in MkGradients layer1WeightGrad layer1BiasGrad layer2WeightGrad layer2BiasGrad

-- Update network parameters using gradients (gradient descent)
export
updateNetwork : Double  -- learning rate
             -> TwoLayerNN input hidden output
             -> Gradients input hidden output
             -> TwoLayerNN input hidden output
updateNetwork learningRate nn grads =
  let newLayer1 = MkFCLayer
        (mSub nn.layer1.weights (mScale learningRate grads.layer1WeightGrad))
        (vSub nn.layer1.biases (scale learningRate grads.layer1BiasGrad))
      newLayer2 = MkFCLayer
        (mSub nn.layer2.weights (mScale learningRate grads.layer2WeightGrad))
        (vSub nn.layer2.biases (scale learningRate grads.layer2BiasGrad))
  in MkTwoLayerNN newLayer1 newLayer2

-- Single training step on one example
export
trainStep : {inp, hid, out : Nat}
         -> Double  -- learning rate
         -> TwoLayerNN inp hid out
         -> (Vect inp Double, Vect out Double)  -- (input, target)
         -> TwoLayerNN inp hid out
trainStep lr nn (inputVec, target) =
  let cache = forwardWithCache nn inputVec
      grads = backward nn cache target
  in updateNetwork lr nn grads

-- Initialize a layer with zeros (dummy initialization)
export
initLayer : (input : Nat) -> (output : Nat) -> FCLayer input output
initLayer input output = MkFCLayer (replicate output (replicate input 0.0)) (replicate output 0.0)

-- Initialize a two-layer network with zeros
export
initTwoLayerNN : (input : Nat) -> (hidden : Nat) -> (output : Nat) -> TwoLayerNN input hidden output
initTwoLayerNN input hidden output =
  MkTwoLayerNN (initLayer input hidden) (initLayer hidden output)
