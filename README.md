# C2_moral_prediction 
The module contains the function (*textpreprocessed2moral*) that evaluates the moral dimension of Twitter posts in Italian about immigration [[1]](#1). Namely, limited to the immigration subject, the model is capable to classify tweets according to the expression of both moral dyads:

| Id | Moral dyad           |
|----|----------------------|
| 0  | Care/Harm            |
| 1  | Fairness/Cheating    |
| 2  | Loyalty/Betrayal     |
| 3  | Authority/Subversion |
| 4  | Purity/Degradation   |
| 5  | No moral             |

and focus concerns:

| Id | Focus concern                            |
|----|------------------------------------------|
| 0  | Prescriptive (if it highlights a virtue) |
| 1  | Prohibitive  (if it blames misbehaviour) |
| 2  | No focus                                 |

The annotation schema is retrieved from [[2]](#2), together with a dataset of 1,724 immigration-related tweets used to fine-tune a pre-trained BERT model for both the tasks [[3]](#3).
Access to the fine-tuned model is provided through the *HuggingFace* platform [https://huggingface.co/brema76/moral_immigration_it].

The module takes as input the output dictionary of the C0_text_preprocessing component. Namely, it first receives all the preprocessed versions of an input tweet whose topic has to be detected. Then it selects only the value of the item corresponding to the key *moral*. The output consists of four lists: the former two concern the labels and the probability vector from the softmax layer for the moral dyads, respectively; the latter two concern the labels and the probability vector from the softmax layer for the focus concerns, respectively.
The fine-tuned model can be loaded directly from the HuggingFace hub: 

    from transformers import AutoModel
    model = AutoModel.from_pretrained('brema76/moral_immigration_it', trust_remote_code=True)

The module contains a Flask app that calculates both the moral dyad distribution and the focus concern distribution for the text corresponding to the value of the item identified by the key *moral* in the output dictionary of the C0_text_preprocessing component.

A working example launching a Flask server to listen on port *127.0.0.1:5000* is reported in the *example.py* file.

## References
<a id="1">[1]</a> 
Brugnoli E, Gravino P, Prevedello G (2023). 
VALE workshop at ECAI, (accepted)

<a id="2">[2]</a> 
Stranisci M, De Leonardis M, Bosco C, Patti V (2021). 
The expression of moral values in the twitter debate: a corpus of conversations. 
IJCoL. Italian Journal of Computational Linguistics, 7 (7-1, 2), Accademia University Press, pp 113-132

<a id="3">[3]</a> 
Schweter S (2020).
Italian BERT and ELECTRA models.
Available from: [https://huggingface.co/dbmdz/bert-base-italian-xxl-uncased]
