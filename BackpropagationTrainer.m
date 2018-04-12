classdef BackpropagationTrainer<handle
    
    %These are to be set before calling the train function
    properties
        network;
        epochs = 100;
        learningRate = 0.001;
        batchSize = 0;
        momentum = 0;
        l2Regularization = 0;
        callbackFn;
        XTrain;
        YTrain;
        XTest;
        YTestHistory = [];
        batchCostHistory = [];
        totalCostHistory = [];
        lastCost;
        epoch = 1;       
        status = "";
    end
    methods                      
        
        
        %XTrain: each row corresponds to an input vector; 
        %YTrain: each row corresponds to an input vector output
        function train(self)
            noExamples = size(self.XTrain, 1);
            prevWeightsUpdate = cell(0);
            weightsUpdate = cell(0);
            %Fix the batch size if 0 or greater than the number of training
            %examples (will result in offline training)           
            if self.batchSize == 0 || self.batchSize > noExamples
                self.batchSize = noExamples;            
            end
            
            lastLayerIndex = length(self.network.layers);
            
            %Initialize the vector of previous weights gradient
            for k = 2:lastLayerIndex                
                prevWeightsUpdate{k} = zeros(size(self.network.weights{k}));
            end
            
            for e = 1:self.epochs
                self.epoch = e;
                %Build the current batch
                fromIndex = mod((e - 1) * self.batchSize, noExamples) + 1;
                toIndex = mod((e - 1) * self.batchSize + self.batchSize - 1, noExamples) + 1;
                if toIndex >= fromIndex
                    XBatch = self.XTrain(fromIndex:toIndex, :);
                    YBatch = self.YTrain(fromIndex:toIndex, :);
                else
                    XBatch = [self.XTrain(fromIndex:end, :); self.XTrain(1:toIndex, :)];
                    YBatch = [self.YTrain(fromIndex:end, :); self.YTrain(1:toIndex, :)];
                end
                
                %Calculate the weights gradient              
                [self.lastCost, weightsGradient] = ...
                    self.calculateWeightsGradient(XBatch, YBatch);
                
                %Update the weights taking into account the learning rate
                %and momentum
                %First for the normal layer to layer weights
                for k = 2:lastLayerIndex
                    weightsUpdate{k} = self.learningRate .* weightsGradient{k} ...
                        + self.momentum .* prevWeightsUpdate{k};                   
                    self.network.weights{k} = self.network.weights{k} ...                        
                        - weightsUpdate{k};
                    %self.network.weights{k} = self.network.weights{k} ...
                    %    .* (1 - self.learningRate * self.l2Regularization / size(self.XTrain, 1)) ...
                    %    - weightsUpdate{k};
                end                
                
                %Update the cost history per batches
                self.batchCostHistory = [self.batchCostHistory mean(self.lastCost)];
                                               
                %Update the previous weights gradient
                prevWeightsUpdate = weightsUpdate;
                
                self.status = "LR:" + self.learningRate + " Mom:" + self.momentum ...
                    + " Batch:" + self.batchSize + "/" + size(self.YTrain, 1) ...
                    +  " Epoch:" + self.epoch;
                
                %If a callback function is defined, call it
                %Note: updating the total cost history is slow, should be
                %optional
                if ~isempty(self.callbackFn)
                    Y = self.network.calculateOutput(self.XTrain);
                    self.totalCostHistory = [self.totalCostHistory sum(self.network.costFn(Y, self.YTrain))];
                    self.callbackFn(self)
                end
            end
        end
        
        %This method calculates the weights gradient for a batch of
        %training examples
        
        %Notations for the formulae:
        %Bi^k = the error terms of neuron i on layer k
        %E = the cost function
        %Oi^k = the activation function of neuron i on layer k
        %Si^k = the transfer function of neuron i on layer k
        %Wij^k = the weight of transfer from neuron i on layer k-1 to neuron
        %j on layer k
        %dX/dY = the derivative of X with respect to Y
        function [cost, weightsGrad] = calculateWeightsGradient(self, XTrain, YTrain)
            %Same size as the weights array, stores the calculated gradient
            %of each weight
            weightsGrad = cell(0);          
            %Stores the error terms for each weight
            %Note: the error terms are the part of the formula that gets
            %used on the back layer
            errorTerms = cell(0);          
            
            lastLayerIndex = length(self.network.layers);
            for k = 2:lastLayerIndex
                weightsGrad{k} = zeros(size(self.network.weights{k}));               
            end
            
            output = self.network.calculateOutput(XTrain);
            cost = self.network.costFn(YTrain, output);
            
            %On the last layer the error terms are:
            %Bi^k = (dE/dOi^k) * (dOi^k/dSi^k)
            errorTerms{lastLayerIndex} = self.network.costFnDv(output, YTrain)...
                .* self.network.activationFnDv{lastLayerIndex}(output);
            
            %And the weights gradients are:
            %deltaWij^k = Bi^k * (dSj^k/dWij^k)
            %Note: the transfer function (S) is assumed to be summation, so
            %its derivative with respect to a weight is the output of the
            %neuron from the previous layer transfering its value through
            %that weight
            weightsGrad{lastLayerIndex} = self.network.layers{lastLayerIndex - 1}' * errorTerms{lastLayerIndex};
            
            %%%%%%%Calculations for the inner layers%%%%%%%%%%%%%%%                                       
            for k = lastLayerIndex - 1:-1:2
                %Calculate the error terms
                weights = self.network.weights{k+1};
                %The error terms on an inner layer k are calculated as 
                %Bi^k = dOi^k/dSi^k * 
                %sum(n over all neurons of k+1)Bn^(k+1) * dSn^k/dOi^k
                dv = self.network.activationFnDv{k}(self.network.layers{k});                             
                
                temp = errorTerms{k+1} * weights';
                errorTerms{k} = dv .* temp;
                %Calculate the weights gradient as 
                %deltaWij^k =  Bj^k * (dSj^k/dWij^k)
                weightsGrad{k} = self.network.layers{k-1}' * errorTerms{k};                
            end
        end
    end
end