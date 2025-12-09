# Research Notes

## Architecture Summary

This project implements a sentiment analysis system using deep neural networks. We're building three models with increasing complexity:

**Model A** handles the preprocessing - cleaning text, removing noise, and tokenizing the input. This creates a foundation for the other models.

**Model B** uses a Recurrent Neural Network (RNN) with attention mechanism. RNNs are designed to process sequences like text, remembering important context from earlier words. The attention layer helps the model focus on the most relevant parts of the text when making predictions.

**Model C** uses DistilBERT, a transformer-based model. This is our production model that gets deployed to the web application.

## Why Transformers > RNN for Sentiment

Transformers like DistilBERT outperform RNNs for sentiment analysis for several key reasons:

**Parallel Processing**: RNNs must process text sequentially, word by word. Transformers process all words simultaneously, making them much faster to train and run.

**Better Context Understanding**: The self-attention mechanism in transformers allows the model to directly connect any two words in a sentence, regardless of distance. RNNs struggle with long-range dependencies because information must pass through many steps.

**Pre-trained Knowledge**: DistilBERT comes pre-trained on massive text datasets. It already understands language patterns, grammar, and context. We just need to fine-tune it for sentiment, which requires less data and training time than building an RNN from scratch.

**Accuracy**: In practice, transformer models consistently achieve higher accuracy on sentiment tasks. They better understand nuanced language, sarcasm, and complex sentence structures.

## Dataset Choices

We're using three diverse datasets to train and evaluate our models:

**Amazon Health & Personal Care Reviews**: Product reviews with star ratings. These are valuable because they include real consumer opinions with clear sentiment labels (1-5 stars). The domain-specific vocabulary helps us understand how the models handle product-related language.

**IMDB Movie Reviews**: Longer-form text with more complex language. Movie reviews often include nuanced opinions, sarcasm, and mixed sentiments. This tests how well our models handle sophisticated writing styles.

**Twitter/Sentiment140**: Short, informal text with hashtags, mentions, and emoji. This represents casual, real-time social media language. It's important because it's very different from formal reviews and tests model robustness.

Using multiple datasets helps us build a more generalizable model that works across different text types, lengths, and writing styles. It also lets us compare model performance across domains to understand strengths and weaknesses.
