
classdef NeuralNetwork<handle

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
       weights = cell(0)            
                    
       %Unidimensional cell array of booleans
       %The record at index K is TRUE if layer K has a bias unit
       %in this case the last row of weights outgoing from that
       %layer will store the biases
       %Note: if the bias unit is present on this layer, during the
       %feedforward phase the value of the last neuron will be reset to 1
       %after calculations
       biasPresent = cell(0)
       
       %Unidimensional cell array
       %The record at index k stores the neurons of layer k + 1 as row
       %vectors, including the bias unit if present (on the last position)
       %The neurons will store the values calculated during the last 
       %pass of data through the network
       layers = cell(0)
       
       %Unidimensional cell array
       %The record at index k stores the activation function for the neurons 
       %of layer k
       activationFn = cell(0)
       %Same for the function that computes the derivative of activation
       activationFnDv = cell(0)
              
       %The cost function
       costFn
       
       %The cost function derivative
       costFnDv
    end
    
    methods       
        
        function network = NeuralNetwork(nInputs, biasPresent, costFn, costFnDv)
            network.layers{1} = zeros(1, nInputs); 
            network.biasPresent{1} = biasPresent;
            network.costFn = costFn;
            network.costFnDv = costFnDv;
        end
        
        %Calculates the output of the network given a list of input vectors
        %X is a matrix with input vectors per row (inputDim X noInputs)
        %If a bias unit is present on the input layer, X may or may not
        %include it. If it is not present the function will add it as a
        %column of ones at the end of X
        %The output is a matrix with output vectors per row (outputDim X
        %noInputs)                
        function output = calculateOutput(network, X)         
            layersCount = length(network.layers);
            
            %If the first layer has a bias unit but the input vector has
            %one less column, add ones at the end
            inputSize = size(network.weights{2}, 1);
            if network.biasPresent{1} && size(X, 2) == inputSize - 1
                adjustedX = [X ones(size(X, 1), 1)];
            else 
                adjustedX = X;
            end
            
            %Set the values of the input layer as the given input X
            network.layers{1} = adjustedX;
                        
            %Parse subsequent layers and calculate their output
            for i = 2 : layersCount                                             
                %Transfer to the next layer
                %= Compute the value of each neuron on the next layer   
                %To do - change it to use the transferFn
                transfer = network.layers{i - 1} * network.weights{i};

                %Pass the values through the activation function               
                network.layers{i} = network.activationFn{i}(transfer);

                %If the bias unit is present on this layer, set the last
                %value to 1
                if network.biasPresent{i}
                    network.layers{i}(:, end) = 1;
                end
            end
            %The output of the network is the output of the last layer
            output = network.layers{layersCount};
        end
        
        %Initializes the weights with random uniformly distributed values
        %layers: row vector with layer indices to be initialized; [] for
        %all layers
        %range: [from, to]
        function initWeights(network, layers, range)
            if isempty(range)
                range = [0 1];
            end
            
            if size(layers) == 0
                layers = 1:1:length(network.weights);
            end
            
            %Initialize the normal weights (layer to layer)
            for i = 1 : length(layers)
                %network.weights{layers(1, i)} = (range(1, 2) - range(1,1)) ...
                %    .* rand(size(network.weights{layers(1, i)})) + range(1,1);
                network.weights{layers(1, i)} = ...
                    normrnd(range(1,1), range(1,2), size(network.weights{layers(1, i)}));
            end            
        end       
        
        %Add a layer to the network        
        %nNeuronsInLayer: the number of neurons in this layer        
        %activationFn/Dev: the activation function and its derivative for all neurons in this layer       
        %Must take a matrix as input and return a similar sized
        %matrix with each element = activation value of the corresponding 
        %input element
        function addLayer(network, nNeuronsInLayer, biasPresent, activationFn, activationFnDev)
            
            %init the layer as a zero column vector
            layer = zeros(1, nNeuronsInLayer);
            %determine the position in which the layer will be appended to
            %the cell array and append it
            index = size(network.layers, 2) + 1;
            network.layers{index} = layer;
            network.biasPresent{index} = biasPresent;
            %If we have more than one layer, we need to create the weights
            %matrix from the previous one to the new layer
            if index > 1
                %Find the number of neurons in the previous layer
                prevLayer = network.layers{index - 1};
                nNeuronsInPrevLayer = size(prevLayer, 2);
                
                %Create the weights matrix and append it to the weights
                %cell array
                %The weights of connections from a neuron in the previous
                %layer to all neurons in the current one are placed on row
                layerWeights = zeros(nNeuronsInPrevLayer, nNeuronsInLayer);
                network.weights{index} = layerWeights;
                               
                network.activationFn{index} = activationFn;
                network.activationFnDv{index} = activationFnDev;
            end
        end
    end
end