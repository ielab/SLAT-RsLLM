import math

from .base_model import BaseModel
import requests
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

class OPENAIModel(BaseModel):
    def predict(self, query, source_prompt):
        prompt_input = HumanMessage(content=source_prompt.format(query=query))
        response_token_probs = self.model.generate([[prompt_input]]).generations[0][0].generation_info["logprobs"]["content"][0]["top_logprobs"]
        yes_prob = 0
        no_prob = 0
        for response_token_prob in response_token_probs:
            token = response_token_prob["token"]
            log_prob = response_token_prob["logprob"]
            probability = log_prob
            if token.lower() == "yes":
                yes_prob += probability
                break
            #elif token.lower() == "no":
                #no_prob += probability

        return yes_prob
        # Implement GPT-3_5 prediction logic



class GPT3_5Model(OPENAIModel):
    def load_model(self):
        model_kwargs = {'logprobs': True, 'top_logprobs': 5, "seed": 1, 'top_p': 0}
        chat_model = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=self.temperature,  max_tokens=10, api_key=self.api_key, model_kwargs=model_kwargs)
        self.model = chat_model

class GPT4Model(OPENAIModel):
    def load_model(self):
        model_kwargs = {'logprobs': True, 'top_logprobs': 5, "seed": 1}
        chat_model = ChatOpenAI(model="gpt-4-turbo-0613", temperature=self.temperature, max_tokens=10, api_key = self.api_key, model_kwargs=model_kwargs)
        self.model = chat_model
