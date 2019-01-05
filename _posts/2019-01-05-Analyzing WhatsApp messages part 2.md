# Analyzing WhatsApp messages with Python, (part 2)

In [part 1](link a la parte1) of this blogpost I walked through loading and cleaning my WhatsApp history, so now let's do some NLP and look at the content! 

In case you are not familiar with NLP, here's some basic terminology: 

1. *Document*s: the rows/observations in the dataset. In my case, WhatsApp messages.
2. *Tokens*: smaller units of the document (for example single words) that can be used for numerical comparison across documents.

My goal here is to do some topic modeling using NMF and `sklearn` library. This post is going to be a little more technical than the previous one, but I'll do my best to walk you through it! 

These are (on a very high level) the steps I followed:

1. Creation of documents: combining messages into groups of 5
2. Preprocessing and tokenization: cleaning the documents and creating the tokens
3. TF-IDF: an approach to convert tokens into numerical features for modeling
4. Running the NMF model
5. Naming the topics
6. Analysis

### Disclaimer

There are many different choices to make when it comes to NLP and unsupervised learning, and they are very problem dependent. That's why the decisions I've made for this project may not be the best ones for others. I intend in this post to show and explain the different steps I took, and if they don't work for your project as well as for mine, then hopefully it will at least give you some ideas on where to go next :).

### Creating the documents

We should take a second to think about how we are going to define the documents. This is not a minor decision, because the topics are basically going to be based on which tokens are more frequently used together (I'll get back to this).

When I first tried to use each message as a different document, I came across two problems: first of all, text messages are too short, so it was hard to find clear patterns in the usage of words. But second of all, I noticed that when we text, we tend to send one message broken into several lines. As a consequence, each line by itself may not make perfect sense, which makes it harder to classify it under a certain topic. For example, in this conversation with my friend, the first two lines can be grouped into one message and the second two lines into another one. 

![image-20181228115129567](/Users/lailasprejer/Library/Application Support/typora-user-images/image-20181228115129567.png)

I solved both problems by combining messages in groups of maximum 5 messages:

```python
def groupby_messages(data,n):
    # Sort messages by conversation and time sent
    new_data = data.sort_values(by=['conv_name','date'])
    
    # Group messages in groups of n messages, sent by the same person on the same day
    new_data['group'] = new_data.groupby(['conv_name','date1','name']).cumcount()
    new_data['group'] = new_data['group'].apply(lambda x: np.floor(x/float(n)))
    
    new_data = new_data.groupby(['conv_name','date1','name','group'])['msg'].sum().reset_index()                                
    return new_data

history_clean = groupby_messages(history_clean,5)
```

Once I got the documents ready, I moved on to the creation of tokens.

### Preprocessing and tokenization

*Tokenization* is the process of chopping each document into pieces called *tokens*. We may split them into sentences, words, groups of words, a combination of those, or pretty much anything that we want as long as it makes sense. But if we just chopped documents without any cleaning, we may end up with either redundant or useless tokens. For example, we may want to group 'lol' and 'lmao' under one token. Or we may want to remove tokens such as 'the' or 'a', which may appear in most messages. That is why we preprocess the documents before tokenizing them.

Overall my preprocessing consisted of removing stop words, punctuation, digits, duplicated letters, and combining different laughter expressions into a single one. As for tokens, I used single words.

```python
def custom_tokenizer(text):

    # remove punctuation
    remove_punct = str.maketrans('', '', string.punctuation)
    text = text.translate(remove_punct)

    # remove digits and convert to lower case
    remove_digits = str.maketrans('', '', string.digits)
    text = text.lower().translate(remove_digits)
    
    # remove duplicated letters
    text = re.sub(r'([a-z])\1+', r'\1', text)

    # combine 'jaja' expresions (this is how Argentinians laugh)
    text = re.sub(r'(ja)[ja]*', 'ja', text)

    # tokenize
    tokens = word_tokenize(text)

    # remove stop words
    stop_words = stopwords.words('spanish')
    tokens_stop = [y for y in tokens if y not in stop_words]
    
    return tokens_stop
```

Luckily, `nltk` package has two very useful functions: `word_tokenize` takes care of the tokenization process, and `stopwords` has a list of common words to remove in many languages including Spanish.

### TF-IDF

The next step was counting the tokens' occurrences. I used a TF-IDF approach that not only looks at how common a token is in a document, but also downweights common tokens accross all documents. Given the short length of text messages, I thought this was the best approach.

The `TfidfVectorizer` function from `sklearn` takes care of all the hard work. It even allows you to pass some useful custom parameters:

- Custom tokenizer (the one I defined earlier)
- max_df: the maximum number (or percentage) of times that a token can occur in order to keep it. 
- min_df: the minimum number (or percentage) of times that a token can occur in order to keep it. 

```python
# Customize TF-IDF function
tfidf = TfidfVectorizer(tokenizer=custom_tokenizer, max_df=0.5, min_df=100)

# Apply it to my messages
X = tfidf.fit_transform(history_clean['msg'])
```

Removing words that occur in more than 50% of the documents may look too extreme but, thinking about text messages, it doesn't really affect that many words. In the same way, by removing words with less than 100 occurrences I intended to mostly remove typos that my custom tokenizer didn't fix. 

### NMF

You can find a detailed explanation of NMF [here](link), but for the purpose of this blogpost, I'll try to give an intuition of what is happening behind the scenes. What we are going to do next, is to try to find a number of *topics* such that we can express each document as a linear combination of them (for instance, we could say that a certain WhatsApp message is made of 50% love, 30% laughter and 20% good news). 

But of course, the model doesn't really know what each topic is really about. As far as NMF knows, a topic is just a combination of words' weights. Then it is on us to look at its most important words and try to find what they have in common to name the topic. For example, the model may output a topic made of the words ['love','nice','sweet'] that we can easily identify as a love topic. 

About the number of topics to find, that's also on us. I found that for my problem, 40 topics gave me the best results, but you can try different numbers and see what you get! 

```python
nmf = NMF(n_components=40,random_state=0)
doc_topics = nmf.fit_transform(X)
```

So next, I assigned the most relevant topic to each document and checked its distribution

```python
# Assigning topic number to each document. t is an array with length of as many documents as we have
t = np.argmax(doc_topics,axis=1)

# Looking at number of document in each component
plt.bar(pd.Series(t).unique(),pd.Series(t).value_counts())
```

![image-20190102164608546](/Users/lailasprejer/Library/Application Support/typora-user-images/image-20190102164608546.png)

It is pretty common to have one big miscellaneous topic that gathers all the messages that didn't fall under any other topic, but you probably want to avoid having over 50% of your messages labeled under it. Overall, I thought this topic distribution looked good enough! But of course, the most important part was to see how the topics turned out.

I thought that by looking at the 10 most relevant words of each topic I would get a reasonable idea of its content:

 ```python
# Get all topics. Words are indexed. 
d = nmf.components_

# Get the actual words. The order is the index.
w = tfidf.get_feature_names()

words = []

# Iterate through topics. len(d) = 40
for r in range(len(d)):
    # Create a list with the index of the top 10 important words in a topic. 
    a = sorted([(v,i) for i,v in enumerate(d[r])],reverse=True)[0:10]
    # Use the index to get the actual word and append it to our list.
    words.append([w[e[1]] for e in a])

# Create a dataframe to visualize topics
topics = pd.DataFrame({topic_number: range(40), words:words})
 ```

These are five of the topics with their most important words:

![image-20190102171133458](/Users/lailasprejer/Library/Application Support/typora-user-images/image-20190102171133458.png)

If you understand a little Spanish, you may recognize that topic 5 is about thanking, and topic 6 about wishing happy birthday.

After looking carefully at every one I added a third column to the dataframe with the topics' names, and this is how it looks like:

![image-20190102165356289](/Users/lailasprejer/Library/Application Support/typora-user-images/image-20190102165356289.png)

Of course, the topics are usually not 100% well defined, but overall I was pretty happy with how they turned out!

The only thing left was adding the topic number to my original dataframe to facilitate the analysis. As the output of NMF keeps the documents' order, this was an easy task.

```python
# Add the topic number to the dataframe
history_clean['topic'] = t

# Get the name of the topics
history_clean = history_clean.merge(topics,on='topic')
```

There! The topic modeling is now complete! :clap:

If you are following along and your topics don't look like you'd like them to, you can always play around with all the choices I've made: try a different preprocessing, tokenization, number of topics, max_df and min_df, etc. To get a better sense of which one to try first, it is always useful to actually look at the topics (for example you could see if there are words that appear in many of them or words you can group). 

### Analysis

Now I'm ready to play around and see what my messages are about! As I'm building my [tableau dashboard], I once again export the dataframe to a csv.

```python
# For the visualization, I dropped the messages
history_clean.drop('msg',axis=1).to_csv('model_viz.csv')
```

One good thing about Tableau is how easily we can create pretty graphs like this:

![image-20181229172324728](/Users/lailasprejer/Library/Application Support/typora-user-images/image-20181229172324728.png)

So good news for me, is that I mostly use WhatsApp as a tool to make plans. And even though bad news comes in second place, it is only 12% of all my WhatsApp usage so I'm not that worried about it. 

Lets now see my topics in action with my family! 

![image-20181229173907343](/Users/lailasprejer/Library/Application Support/typora-user-images/image-20181229173907343.png)

I like this a lot because it seems pretty accurate. My mom has one of the bigger 'thanks' and 'asking' percentages, and my brother the ones for 'asking' and 'work'. The Argentinian Grandchildren group was made specifically to meet our grandma so it is all about making plans, as opposed to my extended family who lives in Israel, so meeting them is quite hard.

I had a lot of fun looking at my different conversations and created this [Tableau Public Dashboard](link) to easily go through my chat history. If you like the framework you can download it and fill it with your own data! 

All the code used here is fully available in my [Github](link), feel free to reach me with any questions at sprejerlaila@gmail.com or leave a comment! 