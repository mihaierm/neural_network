classdef CostFunctions
   methods(Static)
       function [totalError, errors] = halfSumOfSquares(outcome, expected)
           errors = ((expected - outcome).^2) / 2;           
           totalError = sum(errors, 2);
       end
       
       %Returns the value of the derivative with respect to the
       %output(activation) at neuron 'index' from the final layer
       function dv = halfSumOfSquaresDv(outcome, expected)
           dv = -(expected - outcome);
       end
   end
end