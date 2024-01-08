classdef NN4f < handle
    properties
        inputSize
        hiddenDimLayer2
        hiddenDimLayer3
        hiddenDimLayer4
        otherWeightsForBackprop
        outputClassSize
        w2
        w3
        w4
        w5
        bestValinLoss
        bestWeights
        delta_w2_old
        delta_w3_old
        delta_w4_old
        delta_w5_old
        % Adam optimizer properties
        m_w2, v_w2
        m_w3, v_w3
        m_w4, v_w4
        m_w5, v_w5
        beta1
        beta2
        epsilon
        % A gradient with respect to the input
        gradInput
        % Gradients of weights
        gradW2, gradW3, gradW4, gradW5
    end



    methods
        function obj = NN4f(inputSize, hiddenDimLayer2, hiddenDimLayer3, hiddenDimLayer4, outputClassSize, m)
            obj.inputSize = inputSize;
            obj.hiddenDimLayer2 = hiddenDimLayer2;
            obj.hiddenDimLayer3 = hiddenDimLayer3;
            obj.hiddenDimLayer4 = hiddenDimLayer4;
            obj.outputClassSize = outputClassSize;

            % He initialization
            obj.w2 = randn(hiddenDimLayer2, inputSize) * sqrt(2 / inputSize);
            obj.w3 = randn(hiddenDimLayer3, hiddenDimLayer2) * sqrt(2 / hiddenDimLayer2);
            obj.w4 = randn(hiddenDimLayer4, hiddenDimLayer3) * sqrt(2 / hiddenDimLayer3);
            obj.w5 = randn(outputClassSize * m, hiddenDimLayer4) * sqrt(2 / hiddenDimLayer4);

            obj.bestValinLoss = inf;
            obj.bestWeights = struct('w2', [], 'w3', [], 'w4', []);
            obj.delta_w2_old = 0;
            obj.delta_w3_old = 0;
            obj.delta_w4_old = 0;
            obj.delta_w5_old = 0;

            %  Adam parameters
            obj.m_w2 = zeros(size(obj.w2));
            obj.v_w2 = zeros(size(obj.w2));
            obj.m_w3 = zeros(size(obj.w3));
            obj.v_w3 = zeros(size(obj.w3));
            obj.m_w4 = zeros(size(obj.w4));
            obj.v_w4 = zeros(size(obj.w4));
            obj.m_w5 = zeros(size(obj.w5));
            obj.v_w5 = zeros(size(obj.w5));

            obj.beta1 = 0.9;
            obj.beta2 = 0.999;
            obj.epsilon = 1e-8;
            obj.otherWeightsForBackprop = struct('z1', [], 'z2', [], 'z3', [], 'z4', [], 'act2', [], 'act3', [], 'act4', []);

        end

        function updateBestValues(obj)
            if ~isempty(obj.bestWeights.w2)
                obj.w2 = obj.bestWeights.w2;
            end
            if ~isempty(obj.bestWeights.w3)
                obj.w3 = obj.bestWeights.w3;
            end
            if ~isempty(obj.bestWeights.w4)
                obj.w4 = obj.bestWeights.w4;
            end
            if ~isempty(obj.bestWeights.w5)
                obj.w5 = obj.bestWeights.w5;
            end

        end


        function nnOutput = forwardPass(obj, x)
            % Layer 1
            [x, z1] = obj.Layer1(x);

            % Layer 2
            [x, z2] = obj.Layer2(x, obj.w2);
            act2 = obj.my_tanh(z2);

            % Layer 3
            [x, z3] = obj.Layer3(x, obj.w3);
            act3 = obj.my_tanh(z3);
            % Layer 4
            [x, z4] = obj.Layer4(x, obj.w4);
            act4 = obj.my_tanh(z4);

            % Layer 5
            [x, z5] = obj.Layer5(x, obj.w5);


            % Reshape the outpu
            nnOutput = reshape(x, [], obj.outputClassSize);

            % Update  property
            obj.otherWeightsForBackprop = struct('z1', z1, 'z2', z2, 'z3', z3, 'z4', z4, 'z5', z5, 'act2', act2, 'act3', act3, 'act4', act4);
        end
        function NewBestValues(obj, currentValLoss)
            if currentValLoss < obj.bestValinLoss
                obj.bestWeights.w2 = obj.w2;
                obj.bestWeights.w3 = obj.w3;
                obj.bestWeights.w4 = obj.w4;
                obj.bestWeights.w5 = obj.w5;
                obj.bestValinLoss = currentValLoss;
            end
        end

        function updateWeightsWithGradients(obj, learningRate)
            % Update each weight
            obj.w5 = obj.w5 - learningRate * obj.gradW5;
            obj.w4 = obj.w4 - learningRate * obj.gradW4;
            obj.w3 = obj.w3 - learningRate * obj.gradW3;
            obj.w2 = obj.w2 - learningRate * obj.gradW2;
        end

          function backpropagate(obj, y, nnOutput)
            dlossbydy = nnOutput - y; %  cross-entropy loss

            % backprop through softmax and dense
            errr = dlossbydy; % softmax derivative  in the dlossbydy

            gradz5=reshape(errr, [], 1);
            gradw5 = gradz5 * obj.otherWeightsForBackprop.act4';

            % backprop through tanh activation and  (Layer 4)
            dthbydz4 = 1 - obj.otherWeightsForBackprop.act4.^2;
            gradz4 = (obj.w5' * gradz5) .* dthbydz4;
            gradw4 = gradz4 * obj.otherWeightsForBackprop.act3';

            % backprop through tanh activation and  (Layer 3)
            dthbydz3 = 1 - obj.otherWeightsForBackprop.act3.^2;
            gradz3 = (obj.w4' * gradz4) .* dthbydz3;
            gradw3 = gradz3 * obj.otherWeightsForBackprop.act2';

            % backprop through tanh activation and the dense
            dthbydz2 = 1 - obj.otherWeightsForBackprop.act2.^2;
            gradz2 = (obj.w3' * gradz3) .* dthbydz2;
            gradw2 = gradz2 * obj.otherWeightsForBackprop.z1';

            % Calculate the gradient with respect to the input
            gradz1 = obj.w2' * (obj.w3' * (obj.w4' * (obj.w5' * reshape(dlossbydy, [], 1))));
            obj.gradInput = gradz1; % Store  gradient


            % Store gradients
            obj.gradW5 = gradw5;
            obj.gradW4 = gradw4;
            obj.gradW3 = gradw3;
            obj.gradW2 = gradw2;

        end

    end

    methods (Access = private)
        function [x_ou, z] = Layer1(obj, x)
        x_ou = reshape(x, [], 1);
        z = x_ou;
        end


    %Layer 2-Dense with tanhH activation
        function [x_ou, z] = Layer2(obj,x,w2)
            z = w2 * x ;
            x_ou = obj.my_tanh(z);

        end


        %Layer 3-Dense
        function [x_ou, z] = Layer3(obj,x, w3)
            z = w3 * x;
            x_ou = obj.my_tanh(z);
        end

        %Layer 4 -
    function [x_ou, z] = Layer4(obj, x, w4)
        z = w4 * x;
        x_ou = z;
    end



    %Layer 5 - Softmax
    function [x_ou, z] = Layer5(obj, x, w5)
        z = w5 * x;
        z_reshaped = reshape(z, [], obj.outputClassSize);
        x_ou = obj.my_softmax(z_reshaped);
    end

        function y = my_tanh(obj,x)
                MAXi = 5;
                MINi = -5;
                x(x > MAXi) = MAXi;
                x(x < MINi) = MINi;
                y = tanh(x);
        end

        function p = my_softmax(obj, x)
            x = x - max(x, [], 2);
            expo_x = exp(x);
            p = expo_x ./ sum(expo_x, 2);
        end


    end
end
