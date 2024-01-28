from .base_model import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
torch.manual_seed(0)
class FALCONModel(BaseModel):
    #super with init function
    def __init__(self):
        super().__init__()  # This calls the __init__ method of BaseModel
        self.model = None
        self.tokenizer = None
        self.temperature = 0
        self.yes_id = None
        self.no_id = None
        self.batch_size = 1
        self.device = "cuda"
        self.max_length=2048

    def load_model(self, model_path, tokenizer_path=None):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path if tokenizer_path is None else tokenizer_path)
        self.yes_id = self.tokenizer.convert_tokens_to_ids("Yes")
        print("Yes id: ", self.yes_id)
        self.no_id = self.tokenizer.convert_tokens_to_ids("No")
        print("No id: ", self.no_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

    def batch_predict(self, query, source_batch_dict):
        encoded_inputs = self.tokenizer.batch_encode_plus(
            [source_batch_dict[source_id].format(query=query) for source_id in source_batch_dict],
            truncation=True,  # Truncate to model's max length
            padding='longest',
            return_tensors='pt',
            max_length=self.max_length,
        )

        input_ids = encoded_inputs["input_ids"].to(self.device)
        attention_mask = encoded_inputs["attention_mask"].to(self.device)

        with torch.no_grad():  # Make sure no gradients are computed
            outputs = self.model(input_ids, attention_mask=attention_mask)
            next_token_logits = outputs.logits[:, -1, :]
            #output_sequences = self.model.generate(input_ids, attention_mask=attention_mask)

        probs = F.softmax(next_token_logits, dim=-1)
        max_prob, max_id = torch.max(probs, dim=-1)
        for i in range(max_id.size(0)):  # Iterate over the batch
            print(f"Item {i}:")
            print("  max_prob: ", max_prob[i].item())
            print("  max_id: ", max_id[i].item())
            print("  token: ", self.tokenizer.decode([max_id[i].item()]))

        yes_probs = probs[:, self.yes_id].tolist()
        no_probs = probs[:, self.no_id].tolist()

        result_dict = {}
        for i, source_id in enumerate(source_batch_dict):
            #generated_text = self.tokenizer.decode(output_sequences[i], skip_special_tokens=True)
            #print(generated_text)
            result_dict[source_id] = yes_probs[i] - no_probs[i]
        #print(result_dict)
        return result_dict

