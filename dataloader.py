import pandas as pd
from sklearn.utils import shuffle 
from amseg.amharicNormalizer import AmharicNormalizer as normalizer
from amseg.amharicRomanizer import AmharicRomanizer as romanizer
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab
import re
import numpy as np

maxlen = 30

def preprocess_fun(text, normalized, transliterated):
    # Remove Punctions: 
    text = re.sub(r'[^\w\s]', '', text)
    if normalized:
        text = normalizer.normalize(text)
    if transliterated:
        text = romanizer.romanize(text) 

    return text 

def label_to_int(label):
    label_dict = {
        'ሀገር አቀፍ ዜና' : 0,
        'መዝናኛ' : 1, 
        'ስፖርት' : 2, 
        'ቢዝነስ' : 3, 
        'ዓለም አቀፍ ዜና' : 4, 
        'ፖለቲካ' : 5
    }
    return label_dict[label] 

def int_to_label(int_label):
    label_dict = ['ሀገር አቀፍ ዜና','መዝናኛ', 'ስፖርት', 'ቢዝነስ', 'ዓለም አቀፍ ዜና', 'ፖለቲካ']
    return label_dict[int_label]

def sent_to_int(sentiment):
    sentiment_dict = {
        'negative' : 0,
        'positive' : 1
    }
    return sentiment_dict[sentiment] 

def clean_sentiment_text(row, options):
    """Removes url, mentions, emoji and uppercase from tweets"""
    if options['lowercase']:
        row = row.lower()

    if options['remove_url']:
        row = re.sub(r"(?:\@|https?\://)\S+", "", row)

    if options['remove_mentions']:
        row = re.sub("@[A-Za-z0-9_]+","", row)

    if options['demojify']:
      emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
      row = re.sub(emoj, '', row)

    return row

def int_to_sent(int_sentiment):
    sentiment_list = ['negative', 'positive'] 
    return sentiment_list[int_sentiment]

def build_vocab(sentences, tokenizer):
    counter = Counter()
    for sentence in sentences: 
        words = tokenizer(sentence)
        counter.update(words)
    
    return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

def tokenize_sentences(sentences : list, vocab, tokenizer, max_len = 30):
    tokenized_sentences = []
    for sentence in sentences: 
        words = tokenizer(sentence)
        word_ids = [vocab[word] for word in words] 
        if len(word_ids) > max_len:
            word_ids = word_ids[:max_len]
        else:
            diff = max_len - len(word_ids)
            def_index = vocab.get_default_index()
            word_ids.extend([def_index]*diff)
        tokenized_sentences.append(word_ids)
    return tokenized_sentences

def clean_df(df):
    """removes null values and resets index"""
    # remove null values from dataset
    df = df.dropna()
    #drop repeated rows (drop rows with similar tweet id)
    df = df.drop_duplicates(subset='tweet_id', keep="first")
    df = df.reset_index(drop=True)
    return df

def load_sentiment_data():
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')
    df_valid = pd.read_csv('data/dev.csv')
    # drop "mixed" sentiment from df train and df_test
    df_train = df_train[df_train["sentiment"].str.contains("mixed")==False]
    df_test = df_test[df_test["sentiment"].str.contains("mixed")==False]

    # drop "neutral" sentiment from df train and df_test
    df_train = df_train[df_train["sentiment"].str.contains("neutral")==False]
    df_test = df_test[df_test["sentiment"].str.contains("neutral")==False]
    df_test = clean_df(df_test)
    df_train = clean_df(df_train)
    df_valid = clean_df(df_valid)

    clean_config = {
        'remove_url': True,
        'remove_mentions': True,
        'lowercase': True,
        'demojify': True
    }
    df_test['tweet'] = df_test['tweet'].apply(clean_sentiment_text, args=(clean_config,))
    df_train['tweet'] = df_train['tweet'].apply(clean_sentiment_text, args=(clean_config,))
    df_valid['tweet'] = df_valid['tweet'].apply(clean_sentiment_text, args=(clean_config,))

    raw_train_data, train_labels = list(df_train['tweet']), list(df_train['sentiment'])
    raw_test_data, test_labels = list(df_test['tweet']), list(df_test['sentiment'])

    train_labels = [sent_to_int(x) for x in train_labels]
    test_labels = [sent_to_int(x) for x in test_labels]

    return raw_train_data, train_labels, raw_test_data, test_labels

def load_news_data():
    data = pd.read_csv('data/Amharic News Dataset.csv')
    data = shuffle(data)
    data = data.dropna()

    # data['headline'] = data['headline'].apply(lambda x: preprocess_fun(x))
    n_data = data[['headline','category']]
    n_data.head()

    all_text, all_labels = list(data['headline']), list(data['category'])
    all_labels = [label_to_int(x) for x in all_labels]

    raw_train_data = all_text[:40_000]
    train_labels = all_labels[:40_000]
    raw_test_data = all_text[40_000:]
    test_labels = all_labels[40_000:]

    return raw_train_data, train_labels, raw_test_data, test_labels

def load_data(data_set, normalized, transliterated):

    if data_set == 'sentiment':
        raw_train_data, train_labels, raw_test_data, test_labels = load_sentiment_data()
    elif data_set == 'news':
        raw_train_data, train_labels, raw_test_data, test_labels = load_news_data()
    else: 
        raise ValueError("Please choose 'sentiment' or 'news' as dataset.")
    
    raw_train_data = [preprocess_fun(x, normalized, transliterated) for x in raw_train_data]
    raw_test_data = [preprocess_fun(x, normalized, transliterated) for x in raw_test_data]

    tokenizer = get_tokenizer('basic_english', language='en')
    # The method vocab and the variable vocab get confused after running once.
    built_vocab = build_vocab(raw_train_data, tokenizer) 
    built_vocab.set_default_index(built_vocab['<unk>'])

    train_data = tokenize_sentences(raw_train_data, built_vocab, tokenizer) 

    test_data = tokenize_sentences(raw_test_data, built_vocab, tokenizer) 

    x_train = np.array(train_data)
    y_train = np.array(train_labels)
    x_val = np.array(test_data)
    y_val = np.array(test_labels)

    return built_vocab, tokenizer, x_train, y_train, x_val, y_val

def test_model(model, sentences, built_vocab, tokenizer, task):
    # Processes the sentences 
    if type(sentences) != list:
        sentences = [sentences] 
    processed_sentences = [preprocess_fun(x) for x in sentences] 
    tokenized_sentences = tokenize_sentences(processed_sentences, built_vocab, tokenizer)
    array_version = np.array(tokenized_sentences)
    results = model.predict(array_version)
    results = np.argmax(results, axis=1)
    if task == 'news':
        result_labels = [int_to_label(result) for result in results]
    else:
        result_labels = [int_to_sent(result) for result in results]

    return result_labels