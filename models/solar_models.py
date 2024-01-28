from .base_model import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
torch.manual_seed(0)
class SOLARModel(BaseModel):
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
        self.passage_len = 1024

    def load_model(self, model_path, tokenizer_path=None):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path if tokenizer_path is None else tokenizer_path)
        self.yes_id = self.tokenizer.convert_tokens_to_ids("Yes")
        print("Yes id: ", self.yes_id)
        self.no_id = self.tokenizer.convert_tokens_to_ids("No")
        print("No id: ", self.no_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'


    def batch_predict(self, query, source_batch_dict, real_docs=None, use_example_query=False):
        if real_docs is not None:
            #first we want to combine the docs, and using format like Doc 1: <doc1> Doc 2: <doc2> Doc 3: <doc3>
            #then we want to make sure that overall that docs will not exceed 200 tokens.
            #if it does, we will truncate the docs to the first 200 tokens
            #if it does not, leave it as it is
            combined_docs_dict = {}
            query_dict = {}
            for source_id in source_batch_dict:
                combined_docs = ""
                if use_example_query:
                    current_docs = real_docs[source_id]["snippets"]
                    query_dict[source_id] = real_docs[source_id]["query"]
                else:
                    current_docs = real_docs[source_id]

                if len(current_docs) == 0:
                    each_passage_len = 0
                else:
                    each_passage_len = self.passage_len // len(current_docs)

                for rank, real_doc in enumerate(current_docs):
                    # Tokenize the current document
                    tokenized_doc = self.tokenizer.tokenize(real_doc)

                    # Truncate the tokenized document if it's longer than each_passage_len
                    if len(tokenized_doc) > each_passage_len:
                        tokenized_doc = tokenized_doc[:each_passage_len]

                        # Convert the tokens back to a string
                        truncated_doc = self.tokenizer.convert_tokens_to_string(tokenized_doc)
                    else:
                        truncated_doc = real_doc

                    # Append the truncated document to the combined string
                    combined_docs += f"Snippet {rank + 1}:\n{truncated_doc}\n"
                if combined_docs == "":
                    combined_docs = "The example query returned no result on this search engine.\n"
                combined_docs_dict[source_id] = combined_docs
            if use_example_query:
                prompt_reformed = [source_batch_dict[source_id].format(query=query, example_query = query_dict[source_id], docs=combined_docs_dict[source_id]) for source_id in source_batch_dict]
            else:
                prompt_reformed = [source_batch_dict[source_id].format(query=query, docs=combined_docs_dict[source_id]) for source_id in source_batch_dict]
        else:
            prompt_reformed = [source_batch_dict[source_id].format(query=query) for source_id in source_batch_dict]

        #print(prompt_reformed)
        encoded_inputs = self.tokenizer.batch_encode_plus(
            prompt_reformed,
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
        #max_prob, max_id = torch.max(probs, dim=-1)
        # for i in range(max_id.size(0)):  # Iterate over the batch
        #     print(f"Item {i}:")
        #     print("  max_prob: ", max_prob[i].item())
        #     print("  max_id: ", max_id[i].item())
        #     print("  token: ", self.tokenizer.decode([max_id[i].item()]))

        yes_probs = probs[:, self.yes_id].tolist()
        no_probs = probs[:, self.no_id].tolist()

        result_dict = {}
        for i, source_id in enumerate(source_batch_dict):
            #generated_text = self.tokenizer.decode(output_sequences[i], skip_special_tokens=True)
            #print(generated_text)
            result_dict[source_id] = yes_probs[i] - no_probs[i]
        #print(result_dict)
        return result_dict

    def generate(self, prompt, query, snippet):
        # truncate the snippet to 256 tokens
        tokenized_doc = self.tokenizer.tokenize(snippet)
        if len(tokenized_doc) > 256:
            tokenized_doc = tokenized_doc[:256]
            truncated_doc = self.tokenizer.convert_tokens_to_string(tokenized_doc)
        else:
            truncated_doc = snippet
        prompt_reformed = prompt.replace("{query}", query).replace("{snippet}", truncated_doc)
        # print(prompt_reformed)
        input_ids = self.tokenizer(prompt_reformed, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            # generate output
            output_sequences = self.model.generate(
                input_ids=input_ids,
                max_length=self.max_length,
                temperature=self.temperature + 0.00000000000001,
                top_k=0,
                top_p=0.9,
                repetition_penalty=1.0,
                do_sample=True,
                num_return_sequences=1,
            )
        text = self.tokenizer.decode(output_sequences[0])
        #remove input text and only output

        text_output = text.split("[/INST]Results:\n")[-1]
        # remove special token
        text_output = text_output.replace("</s>", "")
        return text_output

    def generate_description(self, propmt, query, combined_snippet):
        tokenized_doc = self.tokenizer.tokenize(combined_snippet)
        if len(tokenized_doc) > 512:
            tokenized_doc = tokenized_doc[:512]
            truncated_doc = self.tokenizer.convert_tokens_to_string(tokenized_doc)
        else:
            truncated_doc = combined_snippet
        prompt_reformed = propmt.format(query=query, snippets=truncated_doc)
        input_ids = self.tokenizer(prompt_reformed, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            # generate output
            output_sequences = self.model.generate(
                input_ids=input_ids,
                max_length=self.max_length,
                temperature=self.temperature + 0.00000000000001,
                top_k=0,
                top_p=0.9,
                repetition_penalty=1.0,
                do_sample=True,
                num_return_sequences=1,
            )
        text = self.tokenizer.decode(output_sequences[0])
        # remove input text and only output

        text_output = text.split("[/INST]The description of the user query is:\n")[-1]
        # remove special token
        text_output = text_output.replace("</s>", "")
        return text_output






