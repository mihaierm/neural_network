classdef ActivationFunctions
   
    methods(Static)
        
        %TANH
        function out = tanh2(in)
            out = 1-2./(exp(2.*in)+1);
        end
        
        function dv = tanh2Dv(x)
            dv = (1.0 - x .^2 );
        end
        
        %IDENTITY
        function out = identity(in)
            out = in;
        end
        
        function dv = identityDv(x)
            dv = 1;
        end
        
        %RELU
        function out = relu(in)
           out = in;
           out(in<0) = 0;
        end
        
        function dv = reluDv(x)
            dv = zeros(size(x));
            dv(x>0)=1;
        end
        
        %RELU
        function out = leakyRelu(in)
           out = in;
           out(in<0) = 0.01;
        end
        
        function dv = leakyReluDv(x)
            dv = zeros(size(x)) + 0.01;
            dv(x>0)=1;
        end
        
        %SIGMOID
        function out = sigmoid(in)
           out = 1 ./ (1 + exp(-in)) ;
        end
        
        function dv = sigmoidDv(x)
            dv = x .* (1 - x);
        end
        
        %TANH
        function out = tanh(x)
            out = (exp(x) - exp(-x)) ./ (exp(x) + exp(-x));
        end
        
        function dv = tanhDv(x)
            dv = 1 - (x .^ 2);
        end
        
        %OTHERS
        function out = linear(in)
            out = in;        
        end
        
        %Hard limit function
        function out = hardLimit(in)
            out = zeros(size(in));         
            out(in >= 0) = 1;
        end
        
        %Symmetrical hard limit function
        function out = hardLimitS(in)
            out = zeros(size(in)) - 1;
            out(in >= 0) = 1;
        end                       
    end
    
end