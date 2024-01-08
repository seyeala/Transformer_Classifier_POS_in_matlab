classdef ffnn3f < handle
    properties
        dim1
        dim2
        inputSize
        hiddenDim
        w1
        w2
        bestValinLoss
        bestWeights
        delta_w1_old
        delta_w2_old
        otherWeightsForBackprop
        gradW1, gradW2 % Gradients of weights
        gradInput % Gradient with respect to the input
        m_w1 
        v_w1 
        m_w2 
        v_w2
        beta1 
        beta2
        epsilon
    end
    
    methods
        function obj = ffnn3f(dim1, dim2, hiddenDim)
            obj.dim1 = dim1;
            obj.dim2 = dim2;
            obj.inputSize = dim1 * dim2;
            obj.hiddenDim = hiddenDim;
            
            obj.w1 = (rand(hiddenDim, obj.inputSize) - 0.5) * 0.1;
            obj.w2 = (rand(obj.inputSize, hiddenDim) - 0.5) * 0.1;

            obj.bestValinLoss = inf;
            obj.bestWeights = struct('w1', [], 'w2', []);
            obj.otherWeightsForBackprop = struct('z0',[],'z1', [], 'z2',[], 'act1', [], 'act2',[]);
    
            obj.m_w1 = zeros(size(obj.w1));
            obj.v_w1 = zeros(size(obj.w1));
            obj.m_w2 = zeros(size(obj.w2));
            obj.v_w2 = zeros(size(obj.w2));
            obj.beta1 = 0.9;
            obj.beta2 = 0.999;
            obj.epsilon = 1e-8;
        end
        
        function updateWeightsWithAdam(obj, t)
            obj.m_w1 = obj.beta1 * obj.m_w1 + (1 - obj.beta1) * obj.gradW1;
            obj.v_w1 = obj.beta2 * obj.v_w1 + (1 - obj.beta2) * (obj.gradW1 .^ 2);
    
            m_w1_hat = obj.m_w1 / (1 - obj.beta1^t);
            v_w1_hat = obj.v_w1 / (1 - obj.beta2^t);
    
            obj.w1 = obj.w1 - obj.learningRate * m_w1_hat ./ (sqrt(v_w1_hat) + obj.epsilon);
        end
        
        function nnOutput = forwardPass(obj, x)
            x = reshape(x, obj.inputSize, []);
    
            z1 = obj.w1 * x;
            act1 = obj.my_tanh(z1);
    
            z2 = obj.w2 * act1;
            act2 = obj.my_tanh(z2);
    
            nnOutput = reshape(act2, obj.dim1, obj.dim2);
    
            obj.otherWeightsForBackprop.z0 = x;
            obj.otherWeightsForBackprop.z1 = z1;
            obj.otherWeightsForBackprop.act1 = act1;
            obj.otherWeightsForBackprop.z2 = z2;
            obj.otherWeightsForBackprop.act2 = act2;
        end
        
        function updateWeightsWithGradients(obj, learningRate)
            obj.w1 = obj.w1 - learningRate * obj.gradW1;
            obj.w2 = obj.w2 - learningRate * obj.gradW2;
        end
        
        function updateBestValues(obj)
            obj.w1 = obj.bestWeights.w1;
            obj.w2 = obj.bestWeights.w2 ;
        end
    
        function NewBestValues(obj, avgValLoss)
            if avgValLoss < obj.bestValinLoss
                obj.bestWeights.w1 = obj.w1;
                obj.bestWeights.w2 = obj.w2;
                obj.bestValinLoss = avgValLoss;
            end
        end
        
        function backpropagate(obj, gradNextLayer)
            gradNextLayer = reshape(gradNextLayer, size(obj.otherWeightsForBackprop.act2));
        
            dthbydz2 = 1 - obj.otherWeightsForBackprop.act2.^2;
            gradz2 = gradNextLayer .* dthbydz2;
    
            obj.gradW2 = gradz2 * obj.otherWeightsForBackprop.act1';
            
            dthbydz1 = 1 - obj.otherWeightsForBackprop.act1.^2;
            gradz1 = (obj.w2' * gradz2).*dthbydz1;
    
            obj.gradW1 = gradz1 * obj.otherWeightsForBackprop.z0';
    
            obj.gradInput = obj.w1' * gradz1;
        end
        
        function [x_ou, z] = layer1(obj, x, w1)
            z = w1 * x;
            x_ou = obj.my_tanh(z);
        end
        
        function [x_ou, z] = layer2(obj, x, w2)
            z = w2 * x;
            x_ou = obj.my_tanh(z);
        end
        
        function y = my_tanh(obj, x)
            MAXi = 10;
            MINi = -10;
            x(x > MAXi) = MAXi;
            x(x < MINi) = MINi;
            y = tanh(x);
        end
    end
end
