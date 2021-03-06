
classdef NeuralNetwork<handle

    properties
       %Unidimensional cell array
       %The record at index k corresponds to the weights of transfer from layer
       %k - 1 to layer k (so index 1 is always empty)
       %Each record is a matrix of size neurons(k - 1) X neurons(k)
       %=> Per line: weights from neuron i (back layer, incl. bias) to neurons 1..j
       %(fw layer)
       %=> Per column: weights from neuron j (fw layer) to neurons 1..i
       %(back layer)
       %=> TO TRANSFER, USE: Y = XW
       weights = cell(0)            
                    
       %Unidimensional cell array of booleans
       %The record at index K is TRUE if layer K has a bias unit
       %in this case the first row of weights outgoing from that
       %layer will store the biases       
       biasPresent = cell(0)
       
       %Unidimensional cell array
       %The record at index k stores the neurons of layer k as row
       %vectors, including the bias unit if present (on the first position)
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
            network.biasPresent{1} = biasPresent;
            if(biasPresent)
                network.layers{1} = zeros(1, nInputs + 1); 
            else
                network.layers{1} = zeros(1, nInputs); 
            end                        
            network.costFn = costFn;
            network.costFnDv = costFnDv;
        end
        
        %Add a layer to the network        
        %nNeuronsInLayer: the number of neurons in this layer        
        %biasPresent: TRUE if this layer has a bias unit
        %activationFn/Dev: the activation function and its derivative for all neurons in this layer       
        function addLayer(network, nNeuronsInLayer, biasPresent, activationFn, activationFnDev)
            %determine the position in which the layer will be appended to
            %the cell array
            index = size(network.layers, 2) + 1;
            
            network.biasPresent{index} = biasPresent;
            
            %init the layer as a zero column vector
            if(biasPresent)
                network.layers{index} = zeros(1, nNeuronsInLayer + 1);
            else
                network.layers{index} = zeros(1, nNeuronsInLayer);
            end
                        
            %Find the number of neurons in the previous layer
            prevLayer = network.layers{index - 1};
            nNeuronsInPrevLayer = size(prevLayer, 2);
                
            %Create the weights matrix and append it to the weights
            %cell array
            layerWeights = zeros(nNeuronsInPrevLayer, nNeuronsInLayer);
            network.weights{index} = layerWeights;
                               
            network.activationFn{index} = activationFn;
            network.activationFnDv{index} = activationFnDev;
        end
        
        %Initializes the weights with random normally distributed values
        %layers is a row vector with layer indices to be initialized; [] for
        %all layers        
        %mean, std: parameters of the normal distribution       
        function initWeights(network, layers, mean, std)            
            if size(layers) == 0
                layers = 1:1:length(network.weights);
            end
            
            %Initialize the normal weights (layer to layer)
            for i = 1 : length(layers)
                network.weights{layers(1, i)} = ...
                    normrnd(mean, std, size(network.weights{layers(1, i)}));
            end            
        end       
        
        %Calculates the output of the network given a list of input vectors
        %X is a matrix with input vectors per row
        %The output is a matrix with output vectors per row
        function output = calculateOutput(network, X)         
            %Number of layers in the network
            layersCount = length(network.layers);
                        
            %Set the values of the input layer as the given input X
            network.layers{1} = X;
                        
            %Parse subsequent layers and calculate their output
            for i = 2 : layersCount                                             
                %If there is a bias unit on the previous layer, insert
                %a column of '1' at the beginning of the neurons matrix                 
                %Having it here ensures no bias is added on the final layer
                if network.biasPresent{i - 1}
                    network.layers{i-1} = [ones(size(X, 1), 1) network.layers{i-1}];
                end
                
                %Calculate the weighted summation values on the layer
                transfer = network.layers{i - 1} * network.weights{i};

                %Pass the values through the activation function               
                network.layers{i} = network.activationFn{i}(transfer);               
            end
            %The output of the network is the output of the last layer
            output = network.layers{layersCount};
        end                       

    end
end