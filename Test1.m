clear;
clear all;

Pattern = (-pi:0.1:pi)';
%Pattern = [Pattern/360 (5*sind(Pattern/3.7+.3)+3*sind(Pattern/1.3+.1)+2*sind(Pattern/34.7+.7))/10];
%Pattern = [Pattern/360 sind(Pattern)];
Pattern = [Pattern -0.85.*cos(2.*Pattern).*Pattern.*exp(-(0.6.*Pattern-.4).^2)];
Pattern(:,2) = (Pattern(:,2) - mean(Pattern(:,2))) ./ std(Pattern(:,2));
Pattern(:,1) = Pattern(:,1);

nn = NeuralNetwork(2, true, @CostFunctions.halfSumOfSquares, @CostFunctions.halfSumOfSquaresDv);
%nn.addLayer(11, true, @ActivationFunctions.tanh, @ActivationFunctions.tanhDv);
%nn.addLayer(51, true, @ActivationFunctions.tanh, @ActivationFunctions.tanhDv);
nn.addLayer(40, true, @ActivationFunctions.tanh, @ActivationFunctions.tanhDv);
nn.addLayer(40, true, @ActivationFunctions.tanh, @ActivationFunctions.tanhDv);
%nn.addLayer(50, true, @ActivationFunctions.relu, @ActivationFunctions.relu);
%nn.addLayer(20, true, @ActivationFunctions.relu, @ActivationFunctions.reluDv);
%nn.addLayer(6, true, @ActivationFunctions.leakyRelu, @ActivationFunctions.leakyReluDv);
%nn.addLayer(6, true, @ActivationFunctions.leakyRelu, @ActivationFunctions.leakyReluDv);
%nn.addLayer(6, true, @ActivationFunctions.leakyRelu, @ActivationFunctions.leakyReluDv);
nn.addLayer(1, false, @ActivationFunctions.identity, @ActivationFunctions.identityDv);
nn.initWeights([], [0, 1]);

%figure;

trainer = BackpropagationTrainer;
trainer.network = nn;
trainer.learningRate = 0.0002;
trainer.momentum = 0. ;
trainer.batchSize = 0;
trainer.epochs = 240000;
trainer.callbackFn = @CallbackFunctions.oneOutputPlot;
trainer.XTrain = Pattern(:,1);
trainer.YTrain = Pattern(:,2);
trainer.train();