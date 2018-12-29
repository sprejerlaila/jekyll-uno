---
title:  "Test"
date:   2018-12-10 15:04:23
categories: [jekyll]
tags: [jekyll]
---
# Analyzing WhatsApp messages with Python (part 1)

I'm one of those people who keeps *every* conversation they ever had on WhatsApp. I've never deleted a single one since 2014, when I got my first smartphone. That means, I've got in my WhatsApp history over 800k messages with my friends, coworkers, family and even old boyfriends. 

As soon as I learned NLP techniques, my first target was clear: go through my entire my whatsapp history to understand how my texting has evolved over the years, how my relationships differed from one another and why not, to see what else I could learn about myself! 

You can find my full code and final presentation in my [GitHub repo] (https://github.com/sprejerlaila/whatsapp-me)

In this first part, I'm going to walk through importing conversations, handling emojis, and some general analysis on messages.

In part 2, I'm going to jump right into my WhatsApp topic analysis using NLP techniques.

### Disclaimer 

In this first approach and for simplicity, I decided to skip media files. 

### Loading the message history

WhatsApp's encrypted messaging makes it quite hard to access even our own message history. As a workaround I decided to manually export each conversation and then load it using `re` syntax.

```python
def read_history(file):
    f = open('data/{}'.format(file), 'r')
 
    # Every text message has the same format: date - sender: message. 
    messages = re.findall('(\d+/\d+/\d+, \d+:\d+\d+) - (.*): (.*)', f.read())
    f.close()

    #Convert list to a dataframe and name the columns
    history = pd.DataFrame(messages,columns=['date','name','msg'])
    history['date'] = pd.to_datetime(history['date'],format="%m/%d/%y, %H:%M")
    history['date1'] = history['date'].apply(lambda x: x.date())

    # file is in the format 'WhatsApp Conversation with XXX.txt'
    history['conv_name'] = file[19:-4]

    return history
```

I saved all my conversations in a 'data' folder, so I list, load, and merge them into one dataframe.

```python
# List all files in directory
files = os.listdir('data')

all = []
for file in files:
    history = read_history(file)
    all.append(history)
    
history = pd.concat(all).reset_index()

# Media messages appear as <Media omitted>, so I delete them
history_clean = history[history['msg']!=' <Media omitted>']
```

At the end I've got a pretty dataframe that looks like this:

![image-20181228115119940](/Users/lailasprejer/Library/Application Support/typora-user-images/image-20181228115119940.png)

### First, some statistics

How many messages would you say you sent over the last 5 years? How many people did you talk to? 

In my case, I sent over 350k messages, talked to over 3000 different people (including group chats) and had 474 unique whatsapp conversations.

Basic Pandas syntax will easily give us all three: 

```python
history_clean[history_clean['name']=='Lai']['msg'].count() # Number of messages sent
history_clean['name'].nunique() # Number of people who I talked to
history_clean['conv_name'].nunique() # Number of unique conversations
```

But of course I looked into the number and length of messages I've sent and received over the years. Here are some fun things I encountered:

First, the number of messages I've sent over the years:

```python
# Create a subset of the dataframe with only messages i've sent
msg_lai = history_clean[history_clean['name']=='Lai']

# Plot
msg_lai.groupby(['date1']).count()['msg'].plot()
```



![image-20181226192343210](/Users/lailasprejer/Library/Application Support/typora-user-images/image-20181226192343210.png)

I was very surprised when I saw this, because every time my texting drops I happen to start dating someone:see_no_evil: (That is, during 2014, and between 2016 and mid 2017).

I also found it quite funny to look at the length of my messages:

```python
history_clean.groupby(['date1'])['msg_len'].mean().plot()
```

![image-20181226192937771](/Users/lailasprejer/Library/Application Support/typora-user-images/image-20181226192937771.png)

So of course I looked at that weird outlier and found this:

```python
history_clean[history_clean['msg_len'] == history_clean['msg_len'].max()]['msg'].values
```

![image-20181226193056291](/Users/lailasprejer/Library/Application Support/typora-user-images/image-20181226193056291.png)

Turns out this is a message from a friend wishing me a happy birthday ('feliz cumple' in spanish), sent on March 19th, my birthday :slightly_smiling_face:

But now the funniest part:

### Let's look at the emojis!

Turns out I used a total of 369 different emoijs (I didn't even know that many emojis even existed!) :scream:

This is how I get them: using the emoji library, I create a function to extract emojis from text, apply it to the 'msg' column, and count the unique values. 

```python
def extract_emojis(str):
    return ''.join(c for c in str if c in emoji.UNICODE_EMOJI)

# msg_lai is a subset of history_clean with only sent messages
len(set(msg_lai['msg'].apply(lambda x: extract_emojis(x)).sum()))
```

But now, lets see how my use of emojis changed over time. 

I define a function to count the number of emojis, but I only look at the 50 most common emojis to prevent the dataframe from being too large (there are over 500 emojis!).

```python
def Count_Emojis(df):
    all_words = df['msg'].apply(lambda x: extract_emojis(x)).sum()
    word_count = Counter(all_words)
     
    ordered = {}
    ordered['msg'] = []
    # Create the columns of the new dataframe with the 50 emojis
    for key, number in word_count.most_common()[:50]:
        ordered[key] = []
    
    # Fill in the values with the count of the emojis in each sentence
    for sentence in series:
        sentence_count = Counter(extract_emojis(sentence))
        
        for word in ordered:
            count = sentence_count[word] if sentence_count[word] else 0
            ordered[word] += [count]
    
    # Just add some categorical variables that I'd like to keep for filtering purposes
    ordered['msg'] = list(series)
    ordered['date'] = list(df['date'])
    ordered['name'] = list(df['name'])
    ordered['conv_name'] = list(df['conv_name'])
    
    return pd.DataFrame(ordered)
```

After struggling for a while with different python visualization tools that are not as compatible with emojis as I'd expect, I found that Tableau was pretty easygoing when it comes to emojis, so I exported the dataframe to a CSV and created a [Tableau Dashboard] (link al tableau) that you can download.

I was glad to see that apparently my life is full of love, happiness and surprise! 

![image-20181226202738596](/Users/lailasprejer/Library/Application Support/typora-user-images/image-20181226202738596.png)



And I was amused to see how well my use of emojis represent different moments of my life:

![image-20181226204141043](/Users/lailasprejer/Library/Application Support/typora-user-images/image-20181226204141043.png)

Feel free to take a look and download my [Public Tableau Dashboard] (link), fill it with your own data and play around like me!  

All the code used here is fully available in my [Github] (link), feel free to reach me with any questions at sprejerlaila@gmail.com or leave a comment! 

Also, take a look at [Part 2] (link) for some WhatsApp topic modeling!

