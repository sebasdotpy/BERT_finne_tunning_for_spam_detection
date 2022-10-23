import torch
from config import CONFIG
import numpy as np



# tokenizer = BertTokenizer.from_pretrained(
#     CONFIG["TOKENIZER_PATH"],
#     do_lower_case=True
# )


def preprocess(package, input_text: str):
    """
    Preprocess data before running with model, for example scaling and doing one hot encoding
    :param package: dict from fastapi state including model and preocessing objects
    :param package: string of input to be proprocessed
    :return: list of proprocessed input
    """

    # scale the data based with scaler fit during training
    tokenizer = package['tokenizer']
    input = tokenizer.encode_plus(
                        input_text,
                        add_special_tokens = True,
                        max_length = 32,
                        padding='max_length',
                        return_attention_mask = True,
                        return_tensors = 'pt')

    return input


def predict(package: dict, input: str) -> str:
    """
    Run model and get result
    :param package: dict from fastapi state including model and preocessing objects
    :param input: string of input value
    :return: numpy array of model output
    """

    # We need Token IDs and Attention Mask for inference on the new sentence
    test_ids = []
    test_attention_mask = []

    # process data applaying tokenizer
    encoding = preprocess(package, input)

    # Extract IDs and Attention Mask
    test_ids.append(encoding['input_ids'])
    test_attention_mask.append(encoding['attention_mask'])
    test_ids = torch.cat(test_ids, dim = 0)
    test_attention_mask = torch.cat(test_attention_mask, dim = 0)

    # Forward pass, calculate logit predictions
    model = package['model']
    with torch.no_grad():
        output = model(test_ids.to(CONFIG['DEVICE']), token_type_ids = None, attention_mask = test_attention_mask.to(CONFIG['DEVICE']))

    # get a prediction
    prediction = 'Spam' if np.argmax(output.logits.cpu().numpy()).flatten().item() == 1 else 'Ham'

    return prediction


if __name__=="__main__":
    from model import Model
    from transformers import BertTokenizer
    model = Model
    # model.load_state_dict(
    #     torch.load(CONFIG['MODEL_PATH_PT'], map_location=torch.device(CONFIG['DEVICE']))
    # )
    try:
        model.cuda()
    except:
        pass
    model.eval()

    # add model and other preprocess tools too app state
    package = {
        "tokenizer": BertTokenizer.from_pretrained(
                CONFIG["TOKENIZER_PATH"],
                do_lower_case=True
            ),
        "model": model
    }
    print(predict(package, "Este texto es de prueba"))