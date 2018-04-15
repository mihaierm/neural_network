clear;
clear all;

Pattern = (-pi:0.1:pi)';
Pattern = [Pattern (5*sin(Pattern/3.7+.3)+3*sin(Pattern/1.3+.1)+2*sin(Pattern/34.7+.7))];
%Pattern = [Pattern sin(Pattern)];
%Pattern = [Pattern -0.85.*cos(2.*Pattern).*Pattern.*exp(-(0.6.*Pattern-.4).^2)];
Pattern(:,2) = (Pattern(:,2) - mean(Pattern(:,2))) ./ std(Pattern(:,2));
%Pattern(:,1) = Pattern(:,1);

load sunspot.dat
year=sunspot(:,1); relNums=sunspot(:,2); %plot(year,relNums)
ynrmv=mean(relNums(:)); sigy=std(relNums(:)); 
nrmY=relNums; %nrmY=(relNums(:)-ynrmv)./sigy; 
ymin=min(nrmY(:)); ymax=max(nrmY(:)); 
relNums=2.0*((nrmY-ymin)/(ymax-ymin)-0.5);
% create a matrix of lagged values for a time series vector
Ss=relNums';
idim=10; % input dimension
odim=length(Ss)-idim; % output dimension
for i=1:odim
   y(i)=Ss(i+idim);
   for j=1:idim
       x(i,j) = Ss(i-j+idim); %x(i,idim-j+1) = Ss(i-j+idim);
   end
end
Pattern = [x, y'];

nn = NeuralNetwork(10, true, @CostFunctions.halfSumOfSquares, @CostFunctions.halfSumOfSquaresDv);
%nn.addLayer(11, true, @ActivationFunctions.tanh, @ActivationFunctions.tanhDv);
%nn.addLayer(51, true, @ActivationFunctions.tanh, @ActivationFunctions.tanhDv);
nn.addLayer(5, true, @ActivationFunctions.tanh, @ActivationFunctions.tanhDv);
%nn.addLayer(200, true, @ActivationFunctions.tanh, @ActivationFunctions.tanhDv);
%nn.addLayer(50, true, @ActivationFunctions.relu, @ActivationFunctions.relu);
%nn.addLayer(20, true, @ActivationFunctions.relu, @ActivationFunctions.reluDv);
%nn.addLayer(6, true, @ActivationFunctions.leakyRelu, @ActivationFunctions.leakyReluDv);
%nn.addLayer(6, true, @ActivationFunctions.leakyRelu, @ActivationFunctions.leakyReluDv);
%nn.addLayer(6, true, @ActivationFunctions.leakyRelu, @ActivationFunctions.leakyReluDv);
nn.addLayer(1, false, @ActivationFunctions.identity, @ActivationFunctions.identityDv);
nn.initWeights([], 0, 0.25);

%figure;

trainer = BackpropagationTrainer;
trainer.network = nn;
trainer.learningRate = 0.001;
trainer.momentum = 0.0 ;
trainer.batchSize = 0;
trainer.epochs = 200;
trainer.callbackFn = @CallbackFunctions.oneOutputPlot;
trainer.XTrain = Pattern(:,1:10);
trainer.YTrain = Pattern(:,11);
trainer.train();