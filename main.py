from models import get_BiLSTM_model, get_Transformer_model 
from dataloader import load_data, test_model
import argparse 

def init_params():
    model_type = 'lstm'
    data_set = 'sentiment'
    batch_size = 8
    epochs = 2
    normalized = True
    transliterated = True
    return model_type, data_set, batch_size, epochs, normalized, transliterated

def get_model(model_type, vocab):
    embed_dim = 128  # Embedding size for each token
    num_heads = 3  # Number of attention heads
    ff_dim = 128  # Hidden layer size in feed forward network inside transformer
    maxlen = 30
    if model_type == 'lstm':
        model = get_BiLSTM_model(vocab)
    elif model_type == 'transformer':
        model = get_Transformer_model(vocab, maxlen, embed_dim, num_heads, ff_dim)
    else:
        raise ValueError("Please choose model type from (transformer/lstm)")

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Type of model architecture (transformer/lstm)')
    parser.add_argument('--dataset', help='Type of dataset (news/sentiment)')
    parser.add_argument('--batchsize', help='Batch Size')
    parser.add_argument('--epoch', help='Training Epochs')
    parser.add_argument('--normalize', help='Whether or not to normalize the input')
    parser.add_argument('--trans', help='Whether or not to transliterate the input')
    args = parser.parse_args()

    model_type, data_set, batch_size, epochs, normalized, transliterated = init_params()
    if args.model:
        model_type = args.model
    if args.dataset:
        data_set = args.dataset
    if args.batchsize:
        batch_size = int(args.batchsize)
    if args.epoch:
        epochs = int(args.epoch)
    if args.normalize:
        normalized = True if int(args.normalize) == 1 else False 
    if args.trans:
        transliterated = True if int(args.trans) == 1 else False

    built_vocab, tokenizer, x_train, y_train, x_val, y_val = load_data(data_set, normalized, transliterated)
    model = get_model(model_type, built_vocab)
    
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    results = model.evaluate(x_val, y_val, verbose=2)

    for name, value in zip(model.metrics_names, results):
        print("%s: %.3f" % (name, value))

    # test_sentece = [
    #     'እስራኤል እና ፍልስጤም በቅርቡ የሰላም ስምምነት ለማካሄድ ቆርጠው ተነስተዋል።',
    #     'የአለም ዋንጫ ማጣሪያ በነገው ዕለት በአዲስ አበባ ከተማ ይጀምራል ተብሎ ይጠበቃል።',
    #     'የቤት ኪራይ ዋጋ ከሰሞኑ ሊጨምር እንደሚችል የአማራ ቤቶች ልማት ጽሕፈት ቤት ዋና ሰብሳቢ ወይዘሮ ለምለም ቢሆነኝ ተናገሩ።',
    #     'ተዋናይ ሰለሞን ቦጋለ በቅርቡ ሰርቶ ያጠናቀቀውን ፊልም በመጪው ቅዳሜ ያስመርቃል ተብሎ ይጠበቃል።'
    # ]

    # results = test_model(model, test_sentece, built_vocab, tokenizer, data_set)
    # for i in range(len(results)):
    #         print(test_sentece[i], results[i])