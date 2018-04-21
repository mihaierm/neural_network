clear;
clear all;

Pattern = (-pi:0.1:pi)';
Pattern = [
    Pattern,...
    (5*sin(Pattern/3.7+.3)+3*sin(Pattern/1.3+.1)+2*sin(Pattern/34.7+.7)), ...
    -0.85.*cos(2.*Pattern).*Pattern.*exp(-(0.6.*Pattern-.4).^2)
];
%Pattern = [Pattern sin(Pattern)];
%Pattern = [Pattern -0.85.*cos(2.*Pattern).*Pattern.*exp(-(0.6.*Pattern-.4).^2)];
Pattern(:,2) = (Pattern(:,2) - mean(Pattern(:,2))) ./ std(Pattern(:,2));
Pattern(:,3) = (Pattern(:,3) - mean(Pattern(:,3))) ./ std(Pattern(:,3));
%Pattern(:,1) = Pattern(:,1);

nn = NeuralNetwork(1, true, @CostFunctions.halfSumOfSquares, @CostFunctions.halfSumOfSquaresDv);
%nn.addLayer(11, true, @ActivationFunctions.tanh, @ActivationFunctions.tanhDv);
%nn.addLayer(51, true, @ActivationFunctions.tanh, @ActivationFunctions.tanhDv);
%nn.addLayer(40, true, @ActivationFunctions.tanh, @ActivationFunctions.tanhDv);
nn.addLayer(40, true, @ActivationFunctions.tanh, @ActivationFunctions.tanhDv);
%nn.addLayer(50, true, @ActivationFunctions.relu, @ActivationFunctions.relu);
%nn.addLayer(20, true, @ActivationFunctions.relu, @ActivationFunctions.reluDv);
%nn.addLayer(6, true, @ActivationFunctions.leakyRelu, @ActivationFunctions.leakyReluDv);
%nn.addLayer(6, true, @ActivationFunctions.leakyRelu, @ActivationFunctions.leakyReluDv);
%nn.addLayer(6, true, @ActivationFunctions.leakyRelu, @ActivationFunctions.leakyReluDv);
nn.addLayer(2, false, @ActivationFunctions.identity, @ActivationFunctions.identityDv);
nn.initWeights([], 0, 1);

trainer = BackpropagationTrainer(nn, 8000, Pattern(:,1), Pattern(:,2:3), ... 
    0.01, 0.1, 0, true, 0, @CallbackFunctions.oneOutputPlot, 10 ...
);
trainer.train();
