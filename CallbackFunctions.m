classdef CallbackFunctions
   methods(Static)
        function oneOutputPlot(trainer)
            if mod(trainer.epoch, 10) ~= 0, return, end;            
            subplot(2, 1, 1);          
            hold on
            cla
            plot(trainer.XTrain, trainer.YTrain);
            plot(trainer.XTrain, trainer.network.calculateOutput(trainer.XTrain));
            
            %l = size(trainer.network.layers, 2);
            %for i = 1:size(trainer.network.layers{l-1}, 2)
                 %plot(1:1:size(trainer.YTrain, 1), trainer.network.layers{l-1}(:,i) * trainer.network.weights{l}(i, 1), '--', 'Color', [0,0,0]);
            %     plot(1:1:size(trainer.YTrain, 1), trainer.network.layers{l-1}(:,i), '--', 'Color', [0,0,0]);
            %end
            text(0.1, 0.1, trainer.status + " Cost: " + num2str(trainer.totalCostHistory(1, end)));
            title("Pattern vs. fit, cost = half sum of squares")
            xlabel("Input")
            ylabel("Output")
            %ylim([-1.1 1.1]);
            hold off
            disp("Epoch: " + trainer.epoch + " Cost:" + num2str(trainer.totalCostHistory(1, end)));
            subplot(2, 1, 2);
            plot(trainer.batchCostHistory);
            hold on
            plot(trainer.totalCostHistory);
            title("Total and batch costs");
            xlabel("Epoch");
            ylabel("Cost");
            hold off
            drawnow();
        end
        
   end
end