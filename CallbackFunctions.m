classdef CallbackFunctions
   methods(Static)
        function oneOutputPlot(trainer)            
            subplot(2, 1, 1);          
            hold on
            cla
            dataSize = size(trainer.YTrain,1);
            plot(1:dataSize, trainer.YTrain);
            plot(1:dataSize, trainer.lastOutput);                       
            title("Pattern vs. fit");
            xlabel(trainer.status);
            ylabel("Output")
            hold off            
            subplot(2, 1, 2);
            plot(trainer.batchCostHistory);
            hold on;
            %plot(trainer.totalCostHistory);
            title("Total and batch costs");
            xlabel("Epoch");
            ylabel("Cost");
            hold off;
            drawnow();
        end                      
   end
end