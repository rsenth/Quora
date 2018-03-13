
# Quora duplicate question identification

Quora is a question and answer website, similar to stack exchange. Often, people ask the same question worded differently many times. Identifying these duplicate questions and merging them improves the user experience as the reader can read all the answers in one place.

##### 1. "How can I be a good geologist?"
##### 2. "What should I do to be a great geologist?"
##### 3. "What does it mean that every time I look at the clock the numbers are the same?"
##### 4. "How many times a day do a clock's hands overlap?"

In the four questions above, questions 1) and 2) are duplicates whereas 3) and 4) are not. 

This code uses Siamese network architecture with LSTM layers and frozen word embedding from FastText to calculate the probability if two questions are duplicate. 



