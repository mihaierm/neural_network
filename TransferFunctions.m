classdef TransferFunctions
    methods(Static)
        function out = summation(neurons, weights)
            out = neurons * weights;
        end
        
        %Returns the value of the derivative with respect to the weight
        function dv = summationDv(neurons, weights)
            dv = zeros(size(weights));
            for i = 1:size(dv, 1)
                dv(i, :) = neurons(i);
            end
        end
        
        function dv = summationDvByWeights(fromNeurons, weights, toNeurons)
            dv = zeros(size(weights));
            for i = 1:size(dv, 1)
                dv(i, :) = fromNeurons(i);
            end
        end
    end
end