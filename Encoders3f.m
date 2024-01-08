classdef Encoders3f < handle
    properties
        myObjectsMHA
        myObjectsFFNN
        classifier
        intermediateValues  % store intermediate  values from the forward
    end

    methods
        function obj = Encoders3f(myobjectsMHA, myobjectsFFNN, classifier)
            % constructor to initialize  the properties
            obj.myObjectsMHA = myobjectsMHA;
            obj.myObjectsFFNN = myobjectsFFNN;
            obj.classifier = classifier;
            obj.intermediateValues = struct();  % initialize
        end

        function out = one_forward(obj, inp)
            % one_forward: performs  a forward pass through the NNS

            obj.intermediateValues = struct();  % reset intermediate values for  forward
            numEncoders = min(length(obj.myObjectsMHA), length(obj.myObjectsFFNN));

            for i = 1:numEncoders
                inp0 = inp;
                inp = obj.myObjectsMHA{i}.forwardPass(inp);

                original_shape_mha = size(inp);
                inp = reshape(inp, [], 1);
                inp0 = reshape(inp0, [], 1);

                obj.intermediateValues.mhaOutput{i} = inp;

                inp = obj.apply_residual_connection(inp0, inp);
                gamma = ones(1, size(inp, 1))';
                beta = zeros(1, size(inp, 1))';  %

                epsilon = 2e-6;
                [inp, mu, sigma] = obj.apply_layer_normalization(inp, gamma, beta, epsilon);
                obj.intermediateValues.layerNormValuesMHA{i} = struct('mu', mu, 'sigma', sigma, 'gamma', gamma, 'beta', beta, 'inp', inp);

                inp = reshape(inp, original_shape_mha);

                inp0 = inp;
                inp = obj.myObjectsFFNN{i}.forwardPass(inp);

                original_shape_ffnn = size(inp);
                inp = reshape(inp, [], 1);
                inp0 = reshape(inp0, [], 1);

                obj.intermediateValues.ffnnOutput{i} = inp;

                inp = obj.apply_residual_connection(inp0, inp);
                [inp, mu, sigma] = obj.apply_layer_normalization(inp, gamma, beta, epsilon);
                obj.intermediateValues.layerNormValuesFFNN{i} = struct('mu', mu, 'sigma', sigma, 'gamma', gamma, 'beta', beta, 'inp', inp);

                inp = reshape(inp, original_shape_ffnn);
            end

            out = obj.classifier.forwardPass(inp);
        end

        function gradient = one_backward(obj, nnOutput, targetOutput)
            % one_backward: performs a backward pass through the NNS
            % backpropagate through the classifier
            obj.classifier.backpropagate(targetOutput, nnOutput);
            gradient = obj.classifier.gradInput;

            numEncoders = min(length(obj.myObjectsMHA), length(obj.myObjectsFFNN));

            for i = numEncoders:-1:1
                original_shape = size(gradient);

                % backpropagate through layer normalization and residual connection
                layerNormValues = obj.intermediateValues.layerNormValuesFFNN{i};

                gradient = reshape(gradient, [], 1);

                gradient = obj.backprop_layer_normalization(gradient, layerNormValues);

                gradient = reshape(gradient, original_shape);

                % backpropagate through FFNN
                obj.myObjectsFFNN{i}.backpropagate(gradient);
                gradient = obj.myObjectsFFNN{i}.gradInput;

                original_shape = size(gradient);
                gradient = reshape(gradient, [], 1);

                % backpropagate through layer normalization and residual connection
                layerNormValues = obj.intermediateValues.layerNormValuesMHA{i};
                gradient = obj.backprop_layer_normalization(gradient, layerNormValues);

                gradient = reshape(gradient, original_shape);

                % backpropagate through MHA
                obj.myObjectsMHA{i}.backpropagate(gradient);
                if i > 1
                    gradient = obj.myObjectsMHA{i}.gradInput;
                end
            end
        end
    end

    methods (Access = private)
        function output = apply_residual_connection(~, inp0, inp)
            % apply a residual connection by element-wise adding inp0 and inp
            output = inp0 + inp;
        end

        function [output, mu, sigma] = apply_layer_normalization(~, inp, gamma, beta, epsilon)
            % apply layer normalization along dimension 1
            mu = mean(inp, 1);
            sigma = std(inp, 0, 1);
            output = gamma .* ((inp - mu) ./ (sigma + epsilon)) + beta;
        end

        function gradient = backprop_residual_connection(gradient)
            % gradient is not affected
        end

        function gradient = backprop_layer_normalization(obj, gradient, layerNormValues)
            % extract the normalization parameters
            mu = layerNormValues.mu;
            sigma = layerNormValues.sigma;
            gamma = layerNormValues.gamma;
            beta = layerNormValues.beta;
            epsilon = 1e-6;

            % calculate the gradient for  gamma and beta
            dgamma = sum(gradient .* ((layerNormValues.inp - mu) ./ (sigma + epsilon)), 1);
            dbeta = sum(gradient, 1);

            % calculate the gradient for the input
            d_inp = (gamma .* gradient) ./ (sigma + epsilon);

            % update the gradient  with respect to the input
            gradient = d_inp - mean(d_inp, 1);

            % reshape the gradient to match the  original shape
            gradient = reshape(gradient, size(layerNormValues.inp));

        end
    end
end
