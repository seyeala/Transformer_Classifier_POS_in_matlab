# Transformer_Classifier_POS_in_matlab


POS tagging with Transformer using BASIC functions in MATLAB



1. Data Preparation

Word Vectors: Pre-trained word vectors are loaded from a CSV file (embedding).

Tokenization: Input sentences are tokenized into words using the provided table.

Vectorized: Tokens are vectorized using the word vectors.

Positional Embedding: Positional embedding is added to the vectorized text.

Maximum Token: A maximum number of tokens is selected to limit inputs with tokens greater than this maximum.

One-Hot Shot Embedding: The labels are embedded into four categories.

Datasets: Three datasets (training, validation, and test) are created. Each dataset row includes features (a 2D vector of tokens and one-hot encoded labels for four categories, with POS tags divided into four categories).

2. Model Initialization

Hyperparameters: Various hyperparameters such as d_model (embedding size), maxTokens, learningRate, number of encoders, number of epochs, and more are initialized. The dimensions of the Feed Forward NN of the encoder are also hyper-parameters. The classifier is another NN with two layers. The dimensions of the hidden layers can also be customized.

Multi-Headed Attention and Feedforward Networks: Instances of multi-headed attention and feedforward networks are created in cell arrays (myObjectsMHA and myObjectffnn).

Classifier: A neural network classifier with four output categories is initialized (classifier).

Encoder Stack: An object is created to model all layers (MHA, FFNN, Residual Connections, Layer Normalization, classifier). It is initialized by receiving MHAs, FFNNs, and classifiers. It has two methods, forward and backward. The forward method calculates the output and all intermediates, while the backward method calculates the gradients from the intermediates. All gradients will be stored in the NN objects. Later, using a preferred optimization method (SGD or Adam), the gradients are used for updating the weights.

3. Dataset Preparation

Training, Validation, and Test Sets: The code loads training, validation, and test datasets from CSV files. The dataset structure ensures a maximum number of tokens (maxTokens), and padding is applied when necessary.
4. Training

Training Loop: The code implements a training loop over a specified number of epochs (numEpochs).

Forward Pass: It performs a forward pass through the network (the Encoders object), computes the loss (cross-entropy), and accumulates the total loss. During the feed-forward process, all intermediate weights are stored in the Encoder Classifier, FFNNs, and MHAs objects. These stored values will later be used for gradient calculation of the trainable weights.

Backward Pass: Gradients of the weights are computed and stored in the NN objects. The gradients of inputs are passed to the previous network.

Gradient Update: For each NN (Classifier, FFNNs, and MHAs), their weights are updated using a preferred optimizer.

Validation: Validation loss is calculated and used to update the best model weights.

Update Validation Loss and Best Found Weights: The best-found validation loss is stored in the NNs, and the corresponding weights are saved as the best weights found. Later, at the end of training, these weights can be restored.

5. Model Evaluation

Testing: After training, the code calculates test loss, accuracy, precision, and recall percentages for each category.
6. Requirements Met

The code uses a Transformer-based architecture modified for POS tagging.

Pretrained word vectors are utilized for word embeddings.

Initialization follows He-initialization for weights and initializes biases to 0.

Training is performed with a specified number of epochs.

Both Adam and SGD optimizers are implemented. Only SGD was tested. The results were quite reasonable.

Results include accuracy, precision, and recall for each category.

7. Architecture

The architecture includes word embedding, positional embedding, followed by 4 encoders, and then a classifier with two hidden layers with tanh activation functions and softmax outputting the last layer. Each encoder includes 4 scaled dot product attentions (4 heads of attention), concatenation of the heads, linear transformation of the concatenated heads, residual connection, and layer normalization, a 4 layer feedforward neural network with tanh activation function followed by a residual connection and layer normalization. The output of the stack of 4 encoders is passed to the classifier.

8. Usage Instructions

	Copy all the .m files provided into a folder.

	Open the mainf.m file in MATLAB.

	Change the following line to your embedding file path:

	filePath = '//mistretm/wv.csv'; % Change to your file path

Change these three lines to the training, validation, and test CSV files:

	vectorized = process_text_data2f('/Users/hamdata.csv', maxTokens, h1);
	vectorized2 = process_text_data2f('/Us/Dataset/valid_data.csv', maxTokens, h1);
	vectorized3 = process_text_data2f('/Useads/Dataset/test_data.csv', maxTokens, h1);


Modify the following hyperparameters if needed:

	maxTokens = 12;
	numEpochs = 20; % Number of epochs
	learningRate = 0.0004; % Learning rate

If you desire to change the number of encoders, you can change the number of object-NN as follows:


	classifier.NewBestValues(avgValLoss);
		for i = 1:4
   	 	myObjectffnn{i}.NewBestValues(avgValLoss);
    		myObjectsMHA{i}.NewBestValues(avgValLoss);
	end



Run the MATLAB code.
9. Running the Code

Run the .m file in MATLAB. As it progresses, you'll see the epoch number, validation, and training losses. At the end, it uses the best weights found to have the lowest validation loss, subsequently reporting the test accuracy, recall, F1 score, and precisions.
10. Reported Accuracy

After 20 epochs of training with SGD optimizer, a learning rate of 0.006, and a maximum number of tokens of 12, the following accuracies were achieved:


Feel free to insert the actual accuracy values based on your experiment results.
