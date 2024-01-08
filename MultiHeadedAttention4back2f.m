classdef MultiHeadedAttention4back2f<handle
    properties
        % Weight  matrices
        Wq1, Wk1, Wv1, Wq2, Wk2, Wv2, Wq3, Wk3, Wv3, Wq4, Wk4, Wv4, Wo
        % Bias vectors
        bq1, bk1, bv1, bq2, bk2, bv2, bq3, bk3, bv3, bq4, bk4, bv4, bo
        dk
        numTokens
        d_model
        % Intermediate weights and outputs
        intermediateWeights
        % Best validation loss and corresponding weights
        bestValLoss, bestWeights
        % Gradient of the loss with respect   to the input of this layer
        gradInput
        % Gradients of trainable weights
        gradWq1, gradWk1, gradWv1, gradWq2, gradWk2, gradWv2, gradWq3, gradWk3, gradWv3, gradWq4, gradWk4, gradWv4, gradWo
    end

    methods
        % Constructor to initialize x weights and biases
        function obj = MultiHeadedAttention4back2f(d_model, dk)
            % Initialize properties
            std_dev = sqrt(2 / (d_model * dk));  % He  initialization

            % Initialize  weights for four heads
            obj.Wq1 = std_dev * randn(d_model, d_model);
            obj.Wk1 = std_dev * randn(d_model, d_model);
            obj.Wv1 = std_dev * randn(d_model, d_model);
            obj.Wq2 = std_dev * randn(d_model, d_model);
            obj.Wk2 = std_dev * randn(d_model, d_model);
            obj.Wv2 = std_dev * randn(d_model, d_model);
            obj.Wq3 = std_dev * randn(d_model, d_model);
            obj.Wk3 = std_dev * randn(d_model, d_model);
            obj.Wv3 = std_dev * randn(d_model, d_model);
            obj.Wq4 = std_dev * randn(d_model, d_model);
            obj.Wk4 = std_dev * randn(d_model, d_model);  % Fourth head
            obj.Wv4 = std_dev * randn(d_model, d_model);  % Fourth head
            obj.Wo = std_dev * randn(d_model * 4, d_model);  % Adjust for four heads (linear transformation)

            % Initialize biases for four heads
            obj.bq1 = zeros(1, d_model);
            obj.bk1 = zeros(1, d_model);
            obj.bv1 = zeros(1, d_model);
            obj.bq2 = zeros(1, d_model);
            obj.bk2 = zeros(1, d_model);
            obj.bv2 = zeros(1, d_model);
            obj.bq3 = zeros(1, d_model);
            obj.bk3 = zeros(1, d_model);
            obj.bv3 = zeros(1, d_model);
            obj.bq4 = zeros(1, d_model);
            obj.bk4 = zeros(1, d_model);  % Fourth  head
            obj.bv4 = zeros(1, d_model);  % Fourth head
            obj.bo = zeros(1, d_model);

            obj.dk = dk;
            obj.d_model=d_model;
            % Initialize  properties for  intermediate weights and best
            obj.intermediateWeights = struct('Q1', [], 'K1', [], 'V1', [], 'Q2', [], 'K2', [], 'V2', [], 'Q3', [], 'K3', [], 'V3', [], 'Q4', [], 'K4', [], 'V4', [], 'concatHeads', []);
            obj.bestValLoss = Inf;
            obj.bestWeights = struct('Wq1', [], 'Wk1', [], 'Wv1', [], 'Wq2', [], 'Wk2', [], 'Wv2', [], 'Wq3', [], 'Wk3', [], 'Wv3', [], 'Wq4', [], 'Wk4', [], 'Wv4', [], 'Wo', []);
        end


        function NewBestValues(obj, currentValLoss)
            if currentValLoss < obj.bestValLoss
                % Update best  weights
                obj.bestWeights.Wq1 = obj.Wq1;
                obj.bestWeights.Wk1 = obj.Wk1;
                obj.bestWeights.Wv1 = obj.Wv1;
                obj.bestWeights.Wq2 = obj.Wq2;
                obj.bestWeights.Wk2 = obj.Wk2;
                obj.bestWeights.Wv2 = obj.Wv2;
                obj.bestWeights.Wq3 = obj.Wq3;
                obj.bestWeights.Wk3 = obj.Wk3;
                obj.bestWeights.Wv3 = obj.Wv3;
                obj.bestWeights.Wq4 = obj.Wq4;
                obj.bestWeights.Wk4 = obj.Wk4;
                obj.bestWeights.Wv4 = obj.Wv4;
                obj.bestWeights.Wo = obj.Wo;

                % Update biase
                obj.bestWeights.bq1 = obj.bq1;
                obj.bestWeights.bk1 = obj.bk1;
                obj.bestWeights.bv1 = obj.bv1;
                obj.bestWeights.bq2 = obj.bq2;
                obj.bestWeights.bk2 = obj.bk2;
                obj.bestWeights.bv2 = obj.bv2;
                obj.bestWeights.bq3 = obj.bq3;
                obj.bestWeights.bk3 = obj.bk3;
                obj.bestWeights.bv3 = obj.bv3;
                obj.bestWeights.bq4 = obj.bq4;
                obj.bestWeights.bk4 = obj.bk4;
                obj.bestWeights.bv4 = obj.bv4;
                obj.bestWeights.bo = obj.bo;

                % Update the best  validation los
                obj.bestValLoss = currentValLoss;
            end
        end




        function output = forwardPass(obj, vectorsWithPosition)
            % Extract sizes from iput
            [obj.numTokens, ~] = size(vectorsWithPosition);

            % Apply linear transormation to get querie s, keys, values for each head
            % And add bias terms
            Q1 = vectorsWithPosition * obj.Wq1 + repmat(obj.bq1, obj.numTokens, 1);
            K1 = vectorsWithPosition * obj.Wk1 + repmat(obj.bk1, obj.numTokens, 1);
            V1 = vectorsWithPosition * obj.Wv1 + repmat(obj.bv1, obj.numTokens, 1);

            Q2 = vectorsWithPosition * obj.Wq2 + repmat(obj.bq2, obj.numTokens, 1);
            K2 = vectorsWithPosition * obj.Wk2 + repmat(obj.bk2, obj.numTokens, 1);
            V2 = vectorsWithPosition * obj.Wv2 + repmat(obj.bv2, obj.numTokens, 1);
            Q3 = vectorsWithPosition * obj.Wq3 + repmat(obj.bq3, obj.numTokens, 1);
            K3 = vectorsWithPosition * obj.Wk3 + repmat(obj.bk3, obj.numTokens, 1);
            V3 = vectorsWithPosition * obj.Wv3 + repmat(obj.bv3, obj.numTokens, 1);
            Q4 = vectorsWithPosition * obj.Wq4 + repmat(obj.bq4, obj.numTokens, 1);
            K4 = vectorsWithPosition * obj.Wk4 + repmat(obj.bk4, obj.numTokens, 1);
            V4 = vectorsWithPosition * obj.Wv4 + repmat(obj.bv4, obj.numTokens, 1);

            %  scaled dotproduct atention for  each head
            head1 = obj.scaledDotProductAttention(Q1, K1, V1);
            head2 = obj.scaledDotProductAttention(Q2, K2, V2);
            head3 = obj.scaledDotProductAttention(Q3, K3, V3);
            head4 = obj.scaledDotProductAttention(Q4, K4, V4);

            % concatenate the heads
            concatenatedHeads = [head1, head2, head3, head4];


            % Stor intermediate weights and outputs
            obj.intermediateWeights.Q1 = Q1;
            obj.intermediateWeights.K1 = K1;
            obj.intermediateWeights.V1 = V1;

            obj.intermediateWeights.Q2 = Q2;
            obj.intermediateWeights.K2 = K2;
            obj.intermediateWeights.V2 = V2;

            obj.intermediateWeights.Q3 = Q3;
            obj.intermediateWeights.K3 = K3;
            obj.intermediateWeights.V3 = V3;

            obj.intermediateWeights.Q4 = Q4;
            obj.intermediateWeights.K4 = K4;
            obj.intermediateWeights.V4 = V4;

            obj.intermediateWeights.attentionOutput = concatenatedHeads;

            % Apply final lineartransformation and add  the output bias
            output = concatenatedHeads * obj.Wo + repmat(obj.bo, obj.numTokens, 1);
            obj.intermediateWeights.concatHeads = concatenatedHeads;
        end
        function updateBestValues(obj)
                obj.Wq1=obj.bestWeights.Wq1 ;
                obj.Wk1=obj.bestWeights.Wk1 ;
                obj.Wv1=obj.bestWeights.Wv1 ;

                obj.Wq2=obj.bestWeights.Wq2 ;
                obj.Wk2=obj.bestWeights.Wk2 ;
                obj.Wv2=obj.bestWeights.Wv2 ;

                obj.Wq3=obj.bestWeights.Wq3 ;
                obj.Wk3=obj.bestWeights.Wk3;
                obj.Wv3=obj.bestWeights.Wv3 ;
                obj.Wq4=obj.bestWeights.Wq4 ;
                obj.Wk4=obj.bestWeights.Wk4 ;
                obj.Wv4=obj.bestWeights.Wv4 ;

                obj.Wo=obj.bestWeights.Wo;
        end
        %  scaled dot-product atention
        function attentionOutput = scaledDotProductAttention(obj, Q, K, V)
            % Calculate the dot products of  Q and K^ T
            dotProducts = Q * K';


            scaledDotProducts = dotProducts / sqrt(obj.dk);

            % Apply   softmax to get  attention weights
            attentionWeights = obj.stableSoftmax(scaledDotProducts);

            attentionOutput = attentionWeights * V;
        end


        function s = stableSoftmax(~, x)
            % Subtract the max for  numerical stability
            shiftx = x - max(x, [], 2);
            exps = exp(shiftx);
            s = exps ./ sum(exps, 2);
        end

        function obj = updateWeightsWithGradients(obj, learningRate)
            % Update  weights for each heads query, key, and  value matrices
            obj.Wq1 = obj.Wq1 - learningRate * obj.gradWq1;
            obj.Wk1 = obj.Wk1 - learningRate * obj.gradWk1;
            obj.Wv1 = obj.Wv1 - learningRate * obj.gradWv1;

            obj.Wq2 = obj.Wq2 - learningRate * obj.gradWq2;
            obj.Wk2 = obj.Wk2 - learningRate * obj.gradWk2;
            obj.Wv2 = obj.Wv2 - learningRate * obj.gradWv2;

            obj.Wq3 = obj.Wq3 - learningRate * obj.gradWq3;
            obj.Wk3 = obj.Wk3 - learningRate * obj.gradWk3;
            obj.Wv3 = obj.Wv3 - learningRate * obj.gradWv3;

            obj.Wq4 = obj.Wq4 - learningRate * obj.gradWq4;
            obj.Wk4 = obj.Wk4 - learningRate * obj.gradWk4;
            obj.Wv4 = obj.Wv4 - learningRate * obj.gradWv4;

            obj.Wo = obj.Wo - learningRate * obj.gradWo;
        end

         function obj = backpropagate(obj, gradNextLayer)
            %  gradients for final linear transformation
            % Reshape gradNextLayer to have the same second dimension as Wo
            gradConcatHeads = reshape(gradNextLayer, obj.numTokens, []);

            % Backprop through t he final  linear transformation
            obj.gradWo = obj.intermediateWeights.concatHeads' * gradConcatHeads;
            dConcatHeads = gradConcatHeads * obj.Wo';
            % split the concatenated  head gradient for  each head
            [dHead1, dHead2, dHead3, dHead4] = obj.splitHeads(dConcatHeads, obj.d_model);


            %store gradients  for Q,  K,  V weights for  each head
            obj.gradWq1 = (obj.intermediateWeights.Q1' * dHead1) / obj.numTokens;
            obj.gradWk1 = (obj.intermediateWeights.K1' * dHead1) / obj.numTokens;
            obj.gradWv1 = (obj.intermediateWeights.V1' * dHead1) / obj.numTokens;

            obj.gradWq2 = (obj.intermediateWeights.Q2' * dHead2) / obj.numTokens;
            obj.gradWk2 = (obj.intermediateWeights.K2' * dHead2) / obj.numTokens;
            obj.gradWv2 = (obj.intermediateWeights.V2' * dHead2) / obj.numTokens;

            obj.gradWq3 = (obj.intermediateWeights.Q3' * dHead3) / obj.numTokens;
            obj.gradWk3 = (obj.intermediateWeights.K3' * dHead3) / obj.numTokens;
            obj.gradWv3 = (obj.intermediateWeights.V3' * dHead3) / obj.numTokens;

            obj.gradWq4 = (obj.intermediateWeights.Q4' * dHead4) / obj.numTokens;
            obj.gradWk4 = (obj.intermediateWeights.K4' * dHead4) / obj.numTokens;
            obj.gradWv4 = (obj.intermediateWeights.V4' * dHead4) / obj.numTokens;

            %  gradients from all heads for  the gradient wit respect  to the input
            obj.gradInput = dHead1*obj.Wq1' + dHead1*obj.Wk1' + dHead1*obj.Wv1' + ...
                            dHead2*obj.Wq2' + dHead2*obj.Wk2' + dHead2*obj.Wv2' + ...
                            dHead3*obj.Wq3' + dHead3*obj.Wk3' + dHead3*obj.Wv3' + ...
                            dHead4*obj.Wq4' + dHead4*obj.Wk4' + dHead4*obj.Wv4';

         end
        function [dHead1, dHead2, dHead3, dHead4] = splitHeads(obj, dConcatHeads, headSize)

            idx1 = 1;
            idx2 = idx1 + headSize;
            idx3 = idx2 + headSize;
            idx4 = idx3 + headSize;

            dHead1 = dConcatHeads(:, idx1:idx1+headSize-1);
            dHead2 = dConcatHeads(:, idx2:idx2+headSize-1);
            dHead3 = dConcatHeads(:, idx3:idx3+headSize-1);
            dHead4 = dConcatHeads(:, idx4:end);
        end

    end
end
