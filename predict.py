from utils import *
from model import *
from config import *

if __name__ == '__main__':
    text = '每个糖尿病患者,无论是病情轻重,不论是注射胰岛素,还是口服降糖药,都必须合理地控制饮食。'
    _, word2id = get_vocab()
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    
    # Prepare inputs
    encoding = tokenizer.encode_plus(
        text=list(text),
        add_special_tokens=True,
        return_tensors='pt',
        is_split_into_words=True
    )
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    attention_mask = attention_mask.bool()
    
    # Instantiate and load the model
    model = Model()
    state_dict = torch.load(MODEL_DIR + 'model_epoch_20.pth', map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        y_pred = model(input_ids, attention_mask)
    
    # Handle the output correctly
    label_ids = y_pred[0]
    id2label, _ = get_label()
    labels = [id2label[label_id] for label_id in label_ids]
    
    print(text)
    print(labels)
    
    info = extract(labels, text)
    print(info)
