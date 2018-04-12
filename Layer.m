classdef Layer<handle
    %LAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties        
       %Unidimensional cell array
       %The record at index k corresponds to the weights of transfer from layer
       %k - 1 to layer k (so index 1 is always empty)
       %Each record is a matrix of size neurons(k - 1) X neurons(k)
       %=> Per line: weights from neuron i (back layer) to neurons 1..j
       %(fw layer)
       %=> Per column: weights from neuron j (fw layer) to neurons 1..i
       %(back layer)
       %=> TO TRANSFER, USE: Y = X'W
       %Note: This includes weights of transfer from layer k-1 to
       %the bias unit of layer k, for simplicity. After calculations the
       %last neuron must be reset to 1
       weights;
                   
       %Unidimensional cell array of booleans
       %The record at index K is TRUE if layer K has a bias unit
       %in this case the last row of weights outgoing from that
       %layer will store the biases
       %Note: if the bias unit is present on this layer, during the
       %feedforward phase the value of the last neuron will be reset to 1
       %after calculations
       biasPresent = FALSE;
       
       %Activation function for the neurons        
       activationFn;
       %Function that calculates the derivative of the above
       activationFnDv;
       
       %Transfer function for the neurons 
       transferFn;
       transferFnDv;
       
       %A single value for each layer, storing the layer's dropout probability
       dropoutProb = 0;
       
       %A [1, no_neurons] binary matrix of [0, 1/(1-p)]
       dropout;       
    end
    
    methods
        function obj = Layer(inputs, outputs, activationFn, activationFnDv, dropoutProb, weightsInitFn)            
            obj.weights = weightsInitFn(outputs, inputs);
            obj.activationFn = activationFn;
            obj.activationFnDv = activationFnDv;
            dropoutProb = dropoutProb;
            weightsInitFn = weightsInitFn;
        end
        
        function outputArg = method1(obj,inputArg)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            outputArg = obj.Property1 + inputArg;
        end
    end
end

