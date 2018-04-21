classdef BackpropagationTrainer<handle
    
    %These are to be set before calling the train function
    properties
        %The neural network to train
        network;
        
        %Number of epochs
        epochs = 100;
        
        %The learning rate
        learningRate = 0.001;
        
        %The momentum coefficient
        momentum = 0;
        
        %The batch size. 0 = entire training data
        batchSize = 0;
        
        %If this is true and we are not in offline mode, samples for each
        %batch will be extracted randomly
        randomizeBatch = true;
        
        %L2 regularization
        l2Regularization = 0;
        
        %Training input
        XTrain;
        
        %Training output target
        YTrain;
        
        %Last output on all training data, calculated each
        %updateEachNEpochs
        lastOutput;
        
        %Evolution of the batch cost
        batchCostHistory = [];
        
        %Evolution of the total cost
        totalCostHistory = [];
        
        %After how many epochs to update the total cost
        updateEachNEpochs = 10
        
        %Callback function to call on each update (see
        %updateEachNEpochs)
        callbackFn;      
        
        %The current epoch
        epoch = 1;       
        
        %Info about the training process, updated each
        %updateEachNIterations
        status = "";
    end
    methods                      
        
        function trainer = BackpropagationTrainer(network, epochs, ...
                XTrain, YTrain, ...
                learningRate, momentum, batchSize, randomizeBatch, ...
                l2Regularization, callbackFn, updateEachNEpochs)           
            trainer.network = network;
            trainer.epochs = epochs;
            trainer.XTrain = XTrain;
            trainer.YTrain = YTrain;
            trainer.learningRate = learningRate;
            trainer.momentum = momentum;
            trainer.batchSize = batchSize;
            trainer.randomizeBatch = randomizeBatch;
            trainer.l2Regularization = l2Regularization;
            trainer.callbackFn = callbackFn;      
            trainer.updateEachNEpochs = updateEachNEpochs;
        end
        
        %XTrain: each row corresponds to an input vector; 
        %YTrain: each row corresponds to an input vector's output
        function train(self)
            %Get the number of input records
            noExamples = size(self.XTrain, 1);
            
            %The previous weights update - used for
            %momentum
            prevWeightsUpdate = cell(0);
            
            %The current weight update
            weightsUpdate = cell(0);
            
            %Fix the batch size if 0 or greater than the number of training
            %examples (will result in offline training)           
            if self.batchSize == 0 || self.batchSize > noExamples
                self.batchSize = noExamples;            
            end
            
            %Index of the last layer
            lastLayerIndex = length(self.network.layers);
            
            %Initialize the vector of previous weights update with 0
            %matrices
            for k = 2:lastLayerIndex                
                prevWeightsUpdate{k} = zeros(size(self.network.weights{k}));
            end
            
            %If we are in offline mode, the batches never change
            if noExamples == self.batchSize
                XBatch = self.XTrain;
                YBatch = self.YTrain;
            end
            
            %Train for the specified number of epochs
            for e = 1:self.epochs
                self.epoch = e;
                
                %If not in offline mode then build the current batch. 
                if noExamples ~= self.batchSize
                    %If set to take random samples, do that
                    %Note: sampling WITH replacement
                    if self.randomizeBatch
                        %Create a vector of batchSize random integer indices 
                        %beteen 1 and the number of rows in the training data
                        samples = randi([1, noExamples], 1, self.batchSize);
                        XBatch = self.XTrain(samples,:);
                        YBatch = self.YTrain(samples, :);
                    %Else cycle through the data
                    else
                        fromIndex = mod((e - 1) * self.batchSize, noExamples) + 1;
                        toIndex = mod((e - 1) * self.batchSize + self.batchSize - 1, noExamples) + 1;
                        if toIndex >= fromIndex
                            XBatch = self.XTrain(fromIndex:toIndex, :);
                            YBatch = self.YTrain(fromIndex:toIndex, :);
                        else
                            XBatch = [self.XTrain(fromIndex:end, :); self.XTrain(1:toIndex, :)];
                            YBatch = [self.YTrain(fromIndex:end, :); self.YTrain(1:toIndex, :)];
                        end
                    end
                end
                %Calculate the weights gradient              
                [crtOutput, crtCost, weightsGradient] = ...
                    self.calculateWeightsGradient(XBatch, YBatch);
                
                %Update the weights taking into account the learning rate
                %and momentum                
                for k = 2:lastLayerIndex
                    weightsUpdate{k} = self.learningRate .* weightsGradient{k} ...
                        + self.momentum .* prevWeightsUpdate{k};                   
                    self.network.weights{k} = self.network.weights{k} ...                        
                        - weightsUpdate{k};
                    %L2 regularization disabled for the moment
                    %self.network.weights{k} = self.network.weights{k} ...
                    %    .* (1 - self.learningRate * self.l2Regularization / size(self.XTrain, 1)) ...
                    %    - weightsUpdate{k};
                end                
                
                %Update the cost history per batches
                self.batchCostHistory = [self.batchCostHistory crtCost];
                                               
                %Update the previous weights gradient
                prevWeightsUpdate = weightsUpdate;
                
                %If this is the first, last or multiple of updateEachNEpochs epoch
                %We need to update the total cost
                if self.epoch == 1 || self.epoch == self.epochs ...
                        || mod(self.epoch, self.updateEachNEpochs) == 0
                    
                    %Determine the total cost (necessary only if not in
                    %offline mode) and update the cost history
                    if self.batchSize ~= noExamples
                        self.lastOutput = self.network.calculateOutput(self.XTrain);
                        crtTotalCost = sum(self.network.costFn(self.lastOutput, self.YTrain));                        
                    else
                        self.lastOutput = crtOutput;
                        crtTotalCost = crtCost;
                    end                    
                    self.totalCostHistory = [self.totalCostHistory crtTotalCost];
                    
                    %Update the status text
                    self.status = sprintf("LR:%0.5g Mom:%0.1g Bat:%d/%d Ep:%d Cost:%0.5g",...
                        self.learningRate, self.momentum, self.batchSize,...
                        noExamples, self.epoch, crtTotalCost...
                    );

                    %If a callback function is defined, call it
                    if ~isempty(self.callbackFn)                        
                        self.callbackFn(self)
                    end

                end
                
                
            end
        end
        
        %This method calculates the weights gradient for a batch of
        %training examples. Returns the gradient and the cost.
        
        %Notations for the formulae:
        %Bi^k = the error terms of neuron i on layer k
        %E = the cost function
        %Oi^k = the activation function of neuron i on layer k
        %Si^k = the transfer function of neuron i on layer k
        %Wij^k = the weight of transfer from neuron i on layer k-1 to neuron
        %j on layer k
        %dX/dY = the derivative of X with respect to Y
        function [output, cost, weightsGrad] = calculateWeightsGradient(self, XTrain, YTrain)
            %Same size as the weights array, stores the calculated gradient
            %of each weight
            weightsGrad = cell(0);          
            
            %Stores the error terms for each weight
            %Note: the error terms are the part of the formula that gets
            %used on the back layer
            errorTerms = cell(0);          
            
            %Index of the last layer
            lastLayerIndex = length(self.network.layers);
            %for k = 2:lastLayerIndex
            %    weightsGrad{k} = zeros(size(self.network.weights{k}));               
            %end
            
            output = self.network.calculateOutput(XTrain);
            cost = mean(self.network.costFn(YTrain, output));
            
            %On the last layer the error terms are:
            %Bi^k = (dE/dOi^k) * (dOi^k/dSi^k)
            errorTerms{lastLayerIndex} = self.network.costFnDv(output, YTrain)...
                .* self.network.activationFnDv{lastLayerIndex}(output);
            
            %And the weights gradients are:
            %deltaWij^k = Bi^k * (dSj^k/dWij^k) = Bi^k * Oj
            %Note: the transfer function (S) is assumed to be summation, so
            %its derivative with respect to a weight is the output of the
            %neuron from the previous layer transfering its value through
            %that weight
            prevLayer = self.network.layers{lastLayerIndex - 1};            
            weightsGrad{lastLayerIndex} = prevLayer' * errorTerms{lastLayerIndex}...
                ./ size(errorTerms{lastLayerIndex}, 1);
            
            %%%%%%%Calculations for the inner layers%%%%%%%%%%%%%%%                                       
            for k = lastLayerIndex - 1:-1:2
                %Calculate the error terms
                weights = self.network.weights{k+1};
                %The error terms on an inner layer k are calculated as 
                %Bi^k = dOi^k/dSi^k * 
                %sum(n over all neurons of k+1)Bn^(k+1) * dSn^k/dOi^k
                dv = self.network.activationFnDv{k}(self.network.layers{k});                             
                
                temp = errorTerms{k + 1} * weights';
                
                errorTerms{k} = dv .* temp;
                if self.network.biasPresent{k}
                    errorTerms{k} = errorTerms{k} (:, 2:end);
                end
                
                prevLayer = self.network.layers{k - 1};
                

                %Calculate the weights gradient as 
                %deltaWij^k =  Bj^k * (dSj^k/dWij^k)
                weightsGrad{k} = prevLayer' * errorTerms{k}...
                    ./ size(errorTerms{k}, 1);                
            end
        end
    end
end