#!/usr/bin/env python
# coding: utf-8

# In[3]:


# get_ipython().system('pip install wordcloud')


# # In[4]:


# get_ipython().system('pip install tld') 


# # In[5]:


# get_ipython().system('pip install torch')


# # In[6]:


# get_ipython().system('pip install transformers')


# # In[29]:


# get_ipython().system('pip install googlesearch-python')


# # In[7]:


# get_ipython().system('pip install simpletransformers')


# In[8]:


import re
import os.path
import wordcloud
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# In[71]:


from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
from urllib.parse import urlparse
from tld import get_tld
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel
from torch.nn import functional as F
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import f1_score
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tld import get_tld, is_tld
from tabulate import tabulate


# In[10]:


data = pd.read_csv("malicious_phish.csv")
print(data.shape)
data.head()


# In[11]:


data.info()


# In[12]:


data.isnull().sum()


# In[13]:


data.dtypes


# In[14]:


count = data.type.value_counts()
count


# In[15]:


sns.barplot(x=count.index, y=count)
plt.xlabel('Types')
plt.ylabel('Count')


# In[16]:


# get list of categorical columns
cat_cols = data.select_dtypes(include=['object']).columns.tolist()
cat_cols


# In[17]:


data['url'] = data['url'].replace('www.', '', regex=True) #omit the (www.) from the URL which is a sub domain in itself.
data


# In[18]:


data['Category'] = data['type']
label_encoder = LabelEncoder()
data['Category'] = label_encoder.fit_transform(data['type'])
data.head()


# In[19]:


data.Category


# In[20]:


data


# ## Plotting Wordcloud

# In[21]:


data_benign = data[data.type=='benign']
data_defacement = data[data.type=='defacement']
data_malware = data[data.type=='malware']
data_phishing = data[data.type=='phishing']


# In[22]:


data_benign


# In[23]:


benign_url = " ".join(i for i in data_benign.url)
wordcloud = WordCloud(width=1600, height=800, colormap='Paired', background_color='white').generate(benign_url)
plt.figure(figsize=(12,14))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[24]:


defacement_url = " ".join(i for i in data_defacement.url)
wordcloud = WordCloud(width=1600, height=800, colormap='Paired', background_color='white').generate(defacement_url)
plt.figure(figsize=(12,14))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[25]:


malware_url = " ".join(i for i in data_malware.url)
wordcloud = WordCloud(width=1600, height=800, colormap='Paired', background_color='white').generate(malware_url)
plt.figure(figsize=(12,14))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[26]:


phish_url = " ".join(i for i in data_phishing.url)
wordcloud = WordCloud(width=1600, height=800, colormap='Paired', background_color='white').generate(phish_url)
plt.figure(figsize=(12,14))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## Feature Engineering 

# In[27]:


#Use of IP or not in domain
def have_ip_addr(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        return 1
    else:
        return 0
data['count_use_of_ip'] = data['url'].apply(lambda i: have_ip_addr(i))


# In[28]:


def abnormal_url(url):
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url)
    if match:
        return 1
    else:
        return 0
data['count_abnormal_url'] = data['url'].apply(lambda i : abnormal_url(i))


# In[30]:


data['count.'] = data['url'].apply(lambda i : i.count('.'))
data['count@'] = data['url'].apply(lambda i: i.count('@'))
data['count_https'] = data['url'].apply(lambda i : i.count('https'))
data['count_http'] = data['url'].apply(lambda i: i.count('http'))
data['count%'] = data['url'].apply(lambda i: i.count('%'))
data['count?'] = data['url'].apply(lambda i: i.count('?'))
data['count-'] = data['url'].apply(lambda i: i.count('-'))
data['count='] = data['url'].apply(lambda i: i.count('='))


# In[31]:


def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')
data['count_dir'] = data['url'].apply(lambda i : no_of_dir(i))

def no_of_embed(url):
    urldir = urlparse(url).path
    return urldir.count('//')
data['count_embed'] = data['url'].apply(lambda i : no_of_embed(i))


# In[32]:


def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return 1
    else:
        return 0
data['short_url'] = data['url'].apply(lambda i : shortening_service(i))


# In[33]:


data['type'].value_counts()


# In[34]:


sns.countplot


# In[35]:


data['url_len'] = data['url'].apply(lambda i : len(str(i)))
data['hostname_len'] = data['url'].apply(lambda i : len(urlparse(i).netloc))


# In[36]:


data.head()


# In[37]:


def sus_words(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                      url)
    if match:
        return 1
    else:
        return 0
data['sus_url'] = data['url'].apply(lambda i : sus_words(i))


# In[38]:


data.head()


# In[39]:


#Length of First Directory
def fd_length(url):
    urlpath= urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0
data['fd_length'] = data['url'].apply(lambda i: fd_length(i))


# In[40]:


#Length of Top Level Domain
data['tld'] = data['url'].apply(lambda i: get_tld(i,fail_silently=True))
def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1
data['tld_length'] = data['tld'].apply(lambda i: tld_length(i))


# In[41]:


data=data.drop("tld",1)


# In[42]:


def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits
data['count-digits']= data['url'].apply(lambda i: digit_count(i))

def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters
data['count-letters']= data['url'].apply(lambda i: letter_count(i))


# In[43]:


data.head()


# ## Split data into Training and Testing data

# In[44]:


print (data.columns.tolist())


# In[45]:


train,eva = train_test_split(data,test_size = 0.2)


# In[46]:


data.head()


# # MODEL TRAINING

# In[47]:


# Function to map sentiment labels to encoded values
def map_sentiment_label(label):
    return label_encoder.transform([label])[0]

# Apply the function to the 'sentiment' column in both train and eva DataFrames
train['label'] = train['type'].apply(map_sentiment_label)
eva['label'] = eva['type'].apply(map_sentiment_label)

print(train.shape)


# In[48]:


train_df = pd.DataFrame({
    'text': train['url'][:1500].replace(r'\n', ' ', regex=True),
    'label': train['label'][:1500]
})

eval_df = pd.DataFrame({
    'text': eva['url'][-400:].replace(r'\n', ' ', regex=True),
    'label': eva['label'][-400:]
})


# ## BERT Model

# In[49]:


model_bert = ClassificationModel('bert', 'bert-base-cased', num_labels=4, args={'reprocess_input_data': True, 'overwrite_output_dir': True},use_cuda=False)


# In[50]:


model_bert.train_model(train_df)


# In[51]:


result_bert, model_outputs_bert, wrong_predictions_bert = model_bert.eval_model(eval_df)


# In[52]:


result_bert


# In[53]:


model_outputs_bert


# In[66]:


true_labels_bert = eval_df['label']
predicted_labels_bert = [np.argmax(output) for output in model_outputs_bert]  # Get the predicted labels from the model outputs

accuracy_bert = accuracy_score(true_labels_bert, predicted_labels_bert)
precision_bert = precision_score(true_labels_bert, predicted_labels_bert, average='weighted')
recall_bert = recall_score(true_labels_bert, predicted_labels_bert, average='weighted')
f1_bert = f1_score(true_labels_bert, predicted_labels_bert, average='weighted')

print("Accuracy:", accuracy_bert*100)
print("Precision:", precision_bert*100)
print("Recall:", recall_bert*100)
print("F1 Score:", f1_bert*100)


# In[74]:


confusion_bert = confusion_matrix(true_labels_bert, predicted_labels_bert)


# In[78]:


def plot_confusion_matrix(confusion_matrix, class_names):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',  
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

class_names = ['Benign', 'Defacement', 'Phishing', 'Malware'] 
plot_confusion_matrix(confusion_bert, class_names)


# ## RoBERTa Model

# In[55]:


# Create a RoBERTa model
model_roberta = ClassificationModel('roberta', 'roberta-base', num_labels=4, args={'reprocess_input_data': True, 'overwrite_output_dir': True}, use_cuda=False)


# In[56]:


model_roberta.train_model(train_df)


# In[57]:


# Evaluate the model on the evaluation dataset
result_roberta, model_outputs_roberta, wrong_predictions_roberta = model_roberta.eval_model(eval_df)


# In[67]:


# Calculate accuracy, precision, and recall
true_labels_roberta = eval_df['label']
predicted_labels_roberta = [np.argmax(output) for output in model_outputs_roberta]

accuracy_roberta = accuracy_score(true_labels_roberta, predicted_labels_roberta)
precision_roberta = precision_score(true_labels_roberta, predicted_labels_roberta, average='weighted')
recall_roberta = recall_score(true_labels_roberta, predicted_labels_roberta, average='weighted')
f1_roberta = f1_score(true_labels_roberta, predicted_labels_roberta, average='weighted')

print("Accuracy:", accuracy_roberta*100)
print("Precision:", precision_roberta*100)
print("Recall:", recall_roberta*100)
print("F1 Score:", f1_roberta*100)


# In[79]:


confusion_roberta = confusion_matrix(true_labels_roberta, predicted_labels_roberta)


# In[87]:


def plot_confusion_matrix(confusion_matrix, class_names):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Greens',  
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

class_names = ['Benign', 'Defacement', 'Phishing', 'Malware'] 
plot_confusion_matrix(confusion_roberta, class_names)


# ## XLNet Model

# In[59]:


# Create an XLNet model
model_xlnet = ClassificationModel('xlnet', 'xlnet-base-cased', num_labels=4, args={'reprocess_input_data': True, 'overwrite_output_dir': True}, use_cuda=False)


# In[60]:


model_xlnet.train_model(train_df)


# In[61]:


# Evaluate the model on the evaluation dataset
result_xlnet, model_outputs_xlnet, wrong_predictions_xlnet = model_xlnet.eval_model(eval_df)


# In[68]:


# Calculate accuracy, precision, and recall
true_labels_xlnet = eval_df['label']
predicted_labels_xlnet = [np.argmax(output) for output in model_outputs_xlnet]

accuracy_xlnet = accuracy_score(true_labels_xlnet, predicted_labels_xlnet)
precision_xlnet = precision_score(true_labels_xlnet, predicted_labels_xlnet, average='weighted')
recall_xlnet = recall_score(true_labels_xlnet, predicted_labels_xlnet, average='weighted')
f1_xlnet = f1_score(true_labels_xlnet, predicted_labels_xlnet, average='weighted')

print("Accuracy:", accuracy_xlnet*100)
print("Precision:", precision_xlnet*100)
print("Recall:", recall_xlnet*100)
print("F1 Score:", f1_xlnet*100)


# In[88]:


confusion_xlnet = confusion_matrix(true_labels_xlnet, predicted_labels_xlnet)


# In[94]:


def plot_confusion_matrix(confusion_matrix, class_names):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',  
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

class_names = ['Benign', 'Defacement', 'Phishing', 'Malware'] 
plot_confusion_matrix(confusion_xlnet, class_names)


# ## Results

# In[69]:


# Dummy data for three models
models = ["BERT", "RoBERTa", "XLNet"]
accuracy = [86.75, 85.75, 84.75]
precision = [86.78913003599703, 87.31790796963946, 83.38109650588319]
recall = [86.75, 85.75, 84.75]

# Create a list of lists for the table
data = [
    ["BERT", f"{accuracy[0]:.2f}%", f"{precision[0]:.2f}%", f"{recall[0]:.2f}%"],
    ["RoBERTa", f"{accuracy[1]:.2f}%", f"{precision[1]:.2f}%", f"{recall[1]:.2f}%"],
    ["XLNet", f"{accuracy[2]:.2f}%", f"{precision[2]:.2f}%", f"{recall[2]:.2f}%"],
]

# Create the table using tabulate
table = tabulate(data, headers=["Model", "Accuracy", "Precision", "Recall"], tablefmt="grid")

# Print or display the table
print(table)


# In[70]:


# Data for three models
models = ["BERT", "RoBERTa", "XLNet"]
accuracy = [86.75, 85.75, 84.75]
precision = [86.78913003599703, 87.31790796963946, 83.38109650588319]
recall = [86.75, 85.75, 84.75]

# Set the width of each bar
bar_width = 0.2
index = np.arange(len(models))

# Create subplots for accuracy, precision, and recall
plt.figure(figsize=(6, 4))

# Accuracy bars
plt.bar(index, accuracy, bar_width, label='Accuracy', color='skyblue')

# Precision bars
plt.bar(index + bar_width, precision, bar_width, label='Precision', color='lightgreen')

# Recall bars
plt.bar(index + 2 * bar_width, recall, bar_width, label='Recall', color='lightcoral')

# Labeling and ticks
plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Performance Metrics for Different Models')
plt.xticks(index + bar_width, models)
plt.legend()

# Show the graph
plt.tight_layout()
plt.show()


# In[ ]:




