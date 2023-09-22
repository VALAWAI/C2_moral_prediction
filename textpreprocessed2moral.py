import os
import requests
from flask import Flask, request
from traceback import format_exc
import transformers
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import Dataset, DataLoader

app = Flask(__name__)

class TweetDataset(Dataset):
    def __init__(self, texts, labels_1=None, labels_2=None):
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-uncased")
        self.texts = texts
        self.labels_1 = labels_1 if labels_1 is not None else [0] * len(texts)
        self.labels_2 = labels_2 if labels_2 is not None else [0] * len(texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label_1 = self.labels_1[idx]
        label_2 = self.labels_2[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label_1': torch.tensor(label_1, dtype=torch.long),
            'label_2': torch.tensor(label_2, dtype=torch.long)
        }

def moral_distr(
    text: str
) -> list:
    """Compute the moral distribution for text.
    text : str
        The input text on the immigration subject.

    Returns
    -------
    dict
    """
    moral_dyads = ['care/harm',
                   'fairness/cheating',
                   'loyalty/betrayal',
                   'authority/subversion',
                   'purity/degradation',
                   'no moral'
                   ]
    focus_concerns = ['prescriptive',
                      'prohibitive',
                      'no focus'
                      ]
    
    model = AutoModel.from_pretrained("brema76/moral_immigration_it", trust_remote_code=True)
    text = TweetDataset([text])
    text = DataLoader(text)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.eval()
    probs_1_list = []
    probs_2_list = []
    for batch in text:
        batch_inputs = batch['input_ids'].to(device)
        batch_attention_mask = batch['attention_mask'].to(device)
        outputs = model(batch_inputs, batch_attention_mask)
        logits_1 = outputs[0]
        logits_2 = outputs[1]
        probs_1 = torch.softmax(logits_1, dim=1)
        probs_2 = torch.softmax(logits_2, dim=1)
        probs_1_list.extend(probs_1.tolist())
        probs_2_list.extend(probs_2.tolist())

    return moral_dyads, probs_1_list, focus_concerns, probs_2_list
    
@app.route('/moral_prediction', methods=['GET'])
def get_moral():
    try:
        input_data = request.get_json()
        if "text" not in input_data:
            return {"error": "A dictionary of preprocessed texts is required in the JSON file"}, 400
        if input_data['text']['moral'] is None:
                return {"error": "The input text results to be empty and cannot be processed."}, 400
        text = input_data["text"]['moral']
        moral_prediction = moral_distr(text)
        return {'moral_prediction': moral_prediction}, 200
    except Exception:
        return {"error": format_exc()}, 400
        
if __name__ == '__main__':
    app.run(debug=True)
