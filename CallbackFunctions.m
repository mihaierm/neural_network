classdef CallbackFunctions
   methods(Static)
        function oneOutputPlot(trainer)
            if mod(trainer.epoch, 1) ~= 0, return, end;            
            subplot(2, 1, 1);          
            hold on
            cla
            plot([1:size(trainer.YTrain,1)], trainer.YTrain);
            plot([1:size(trainer.YTrain,1)], trainer.network.calculateOutput(trainer.XTrain));                       
            text(0.1, 0.1, trainer.status + " Cost: " + num2str(trainer.totalCostHistory(1, end)));
            title("Pattern vs. fit")
            xlabel("Input")
            ylabel("Output")
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
        
        
        function twoOutputPlot(trainer)
            if mod(trainer.epoch, 10) ~= 0, return, end;            
            subplot(2, 1, 1);          
            hold on
            cla
            YCalc = trainer.network.calculateOutput(trainer.XTrain);
            [X,Y] = meshgrid(trainer.YTrain(:, 1), trainer.YTrain(:, 2));
            mesh(X, Y, Z);
            
            [X,Y] = meshgrid(YCalc(:, 1), YCalc(:, 2), trainer.XTrain);
            mesh(X);           
            text(0.1, 0.1, trainer.status + " Cost: " + num2str(trainer.totalCostHistory(1, end)));
            title("Pattern vs. fit")
            xlabel("Input")
            ylabel("Output 1")
            ylabel("Output 2")
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