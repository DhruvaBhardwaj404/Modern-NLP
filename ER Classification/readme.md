<h2>For Task 1</h2>
We have a very skewed sft dataset: English has over 50k tuples but hindi and kannada only has about 200. The other challenge
was that even within the english dataset about 80% of the data belongs to just one class label and the label distribution was very skewed with 8-10 labels comprising most of the data.

So first I have tried to solve these challenges by having a maximum cap over how many examples from a particular class can be used for training. I have also set a min limit for labels data, if 
a class has less than min limit tuples it is artificially up-sampled by repeating it in the data. With this every batch has representation from the classes with fewer data, otherwise the model would only 
learn to predict the class with majority data and not the other classes.

For Hindi and Kannada, I have done the same class up-sample technique but after that I am repeating the entire data with a repetition factor so that the model gets to see data for these languages enough times
in the batch to be able to generalise.

For the classification head I have used a simple 2 layer head which predict probabilities for each class. I have not created separate prediction classes for Hindi and Kannada, they are mapped to the corresponding english labels for cross-lingual generalisation which wasn't happening with separate prediction classes.
I have experimented with various ranks and dropout to optimise my model and submitted the best I found through experimentation. 

Since the model is big having a large batch is not possible so I am accumulating gradients over smaller batches before back propagation. The model is also loaded in float16 for memory and speed efficiency.

I have added new symbols to the tokenizer to mark my entities this is the reason why in my Lora Adapter I also have an adapter for embedding layer.
I am pretraining the Lora Adapter on Hindi and Kannada Datasets before SFT. For SFT I am taking final hidden states from three state (end of entity 1, end of entity 2 and end of sentence) and concatenating them and passing as input to classification head.
This captures context from three different points in the sentence, the main reasons is that this way it gets enough information whether the relationship is understood from context before or after.

For SFT I am using weighted Cross Entropy Loss, For calculating weights for each class I am using the english dataset class count and using square root to have smoother weights.

<h2>For Task 2</h2>
Here from data perspective I am again doing the same thing as task 1. I am pretraining in the similar fashion over all the four Indian languages after which I run SFT.
I found that model struggled to generate the class labels in Indian Languages, so I generate the labels in english for all language and even in training it is made to generate english label this made the performance better. This helped in cross-lingual generalisation as without this model's performance struggled a lot.
Then in postprocessing I am changing the labels back to the respective language. 

The generated labels are first compared to find the closest matching label in case some typo occurred and then the corrected label is used as the final prediction. For the prompt I have used chat style template which explicitly marks system and user prompts.

Even in this part I have added entity marking tokens to the tokenizer and so trained lora adapter for embedding layer as well. I found training adapter for all parts of the network better, so I have done that for both the tasks.

I am using warmup with cosine schedule for the training, I am only training for 1 Epoch for both task 1 and task 2.


<h2>For Task 3</h2>
In this I am using FAISS vector database to store examples. I am using MuRIL model for creating embeddings (mean pooling with L2 Normalisation) for the sentences which are then being stored in language wise indices. I am not using all the english examples here instead I am only using 1000 from each class from english at max.
For each query I have drafted a prompt that consist of valid labels model has to predict and along with which I have also added 5 examples from english and 4 from the respective language. All of these examples are being retrieved based on the basis of similarity.

I have tried varying the prompts, I found adding the valid prompt labels to the prompt help it greatly. Also making the model generate labels in english gave better results than otherwise.
I tried varying the temperate and top_p parameter but didn't get good results on my validation set and so decided to set temperature to 0.

I tested adding examples for other Indian languages in each prompt but that led to bad results, so I dropped that idea completely.
