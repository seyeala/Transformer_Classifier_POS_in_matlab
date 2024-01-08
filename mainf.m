
filePath = '/Users/hamid/privacy/admission CS/Collumbia University JAN 2ND/w4995/HW2/mistretm/wv.csv'; % Change to your actual file path
h1 = readvectorsfromcsvf(filePath);

requestedWord = 'start';
vectorForWord = h1('start');

inputString = 'This is test .';
%window for attention or max number of tokens
maxTokens = 12;

d_model = 64;
std_dev = sqrt(2 / d_model);
dk = d_model / 4;
sizeoutput = maxTokens * d_model;

myObjectsMHA = cell(1, 4); % Initialize a cell array of objects, each of them is a Multiheaded attention
myObjectffnn = cell(1, 4); % Initialize a cell array, each of them is a feed forward NN after the MHA

for i = 1:4
    myObjectsMHA{i} = MultiHeadedAttention4back2f(d_model, dk);  % Create an instance of MultiHeadedAttention4back and assign it to the cell array
    myObjectffnn{i} = ffnn3f(maxTokens, d_model, sizeoutput * 2);  % Create an instance of feed forward NN and assign it to the cell array
end
% Create an instance of a classifer NN with 4 categories. This NN will recive the 4 encordes output
classifier = NN4f(maxTokens * d_model, 20, 10, 10, 4, maxTokens);

% Create training validation and test datasets. The cut off length of tokens will be maxTokens
%if the actual number of tokens is less than maxTokens then the toneks and labels will self repeat
%until maxTokens is avaiable.

 vectorized=process_text_data2f('/Users/hamid/Downloads/Dataset/train_data.csv',maxTokens, h1,d_model);
 vectorized2=process_text_data2f('/Users/hamid/Downloads/Dataset/train_data.csv',maxTokens, h1,d_model);
 vectorized3=process_text_data2f('/Users/hamid/Downloads/Dataset/train_data.csv',maxTokens, h1,d_model);

%comment out the next three lines otherwise only a subset of the dataset will be used for a quick run
 vectorized=getSubsetOfDatasetf(vectorized,100)
 vectorized2=getSubsetOfDatasetf(vectorized2,25)
 vectorized3=getSubsetOfDatasetf(vectorized3,25)
% Define training hyperparameters


% Create an instance of the encoders (stack of 4 encoders)
% myObjectsMHA, myObjectffnn, classifier will be instances of the MHA FFNN and classifer
% thaat are part of the encoders
%Layer normalization is seperatly included in the instance
one_forw = Encoders3f(myObjectsMHA, myObjectffnn, classifier);

numEpochs = 90; % Number of epochs
learningRate = 0.0002; % Learning rate
% Loop over epochs
for epoch = 1:numEpochs
    totalLoss = 0; % Initialize total loss for this epoch

    % Shuffle the training dataset at the beginning of each epoch
    shuffledIndices = randperm(length(vectorized));

    % Iterate through each sample in the training dataset
    for idx = 1:length(shuffledIndices)
        i = shuffledIndices(idx); % Get the index of the current sample

        % Get the input and target output
        inputVec = vectorized{i, 1};
        targetOutput = vectorized{i, 2};

        % Forward pass
        nnOutput = one_forw.one_forward(inputVec);

        % Compute loss (assuming cross-entropy loss)
        loss = mean(crossEntropyLoss(targetOutput, nnOutput));
        totalLoss = totalLoss + loss;

        % Backward pass to compute gradients
        one_forw.one_backward(nnOutput, targetOutput);

        % Update the weights using the gradients of all NNs
        classifier.updateWeightsWithGradients(learningRate);
        for i = 1:4
            myObjectffnn{i}.updateWeightsWithGradients(learningRate);
            myObjectsMHA{i}.updateWeightsWithGradients(learningRate);
        end

        %  print batch loss/useful if the dataset is large
        if mod(idx, 1000) == 0
            fprintf('Epoch: %d, Batch: %d, Batch Loss: %.4f\n', epoch, idx, loss / maxTokens);
        end
    end

    %  average loss for the training epoch
    avgLoss = totalLoss / length(vectorized);

    %  validation loss
    avgValLoss = calculateLoss2f(vectorized2, @one_forw.one_forward, myObjectsMHA, myObjectffnn, classifier);

    % Update best validation loss and best weights if the current validation loss is lower
    % than what is stored in the NN properties
    % this way each NN has a track of the best validation loss and the corresponding weights
    % in the end of the training loop, the current weights can be updated with the best-found weights

    classifier.NewBestValues(avgValLoss);
    for i = 1:4
        myObjectffnn{i}.NewBestValues(avgValLoss);
        myObjectsMHA{i}.NewBestValues(avgValLoss);
    end

    % Print epoch results
    fprintf('Epoch: %d, Training Loss: %.4f, Validation Loss: %.4f\n', epoch, avgLoss, avgValLoss);
end

% After training, update all NN with the best weights
% corresponding to the lowest validation loss recorded

classifier.updateBestValues();
for i = 1:4
    myObjectffnn{i}.updateBestValues();
    myObjectsMHA{i}.updateBestValues();
end

fprintf('Updated classifier weights with the best values found during training.\n');

% After training,  calculate test loss, accuracy, precision, and recall percentage for each category

calculateMetricsf(vectorized3, @one_forw.one_forward, myObjectsMHA, myObjectffnn, classifier);
