from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import json
import wandb

defalut_prompt_name_description = \
                 "Federated search retrieves information from a variety of sources via a search application built on top of one or more search engines. A user makes a single query request. The federated search then selects only the search engines that the query should be sent to from a list of search engines, and aggregates the result for presentation of high quality result to the user. The task is called resource selection." \
                 "The following is a search engine with its name, url and description\n\n" \
                 "Name: {name}\n" \
                 "URL: {url}\n" \
                 "Description: {description}\n" \
                 "The following is a real user query: \n" \
                 "{query}\n" \
                 "Now, please reply only yes or no to indicate if the query should be sent to the search engine.\n" \
                 "Response:"


default_prompt_example_snippet = \
                 "Federated search retrieves information from a variety of sources via a search application built on top of one or more search engines. A user makes a single query request. The federated search then selects only the search engines that the query should be sent to from a list of search engines, and aggregates the result for presentation of high quality result to the user. The task is called resource selection." \
                 "The following is a search engine with its name, url and description\n\n" \
                 "Name: {name}\n" \
                 "URL: {url}\n" \
                 "Description: {description}\n" \
                 "The following is a real user query: \n" \
                 "{query}\n" \
                 "The following are some snippets from this search engine that are similar to the user query: \n" \
                 "{snippet}\n" \
                 "Now, please reply only yes or no to indicate if the user query should be sent to the search engine.\n" \
                 "Response:"

default_prompt_name_snippet = \
                 "Federated search retrieves information from a variety of sources via a search application built on top of one or more search engines. A user makes a single query request. The federated search then selects only the search engines that the query should be sent to from a list of search engines, and aggregates the result for presentation of high quality result to the user. The task is called resource selection." \
                 "The following is a search engine with its name and url\n\n" \
                 "Name: {name}\n" \
                 "URL: {url}\n" \
                 "The following is a real user query: \n" \
                 "{query}\n" \
                 "The following are some snippets from this search engine that are similar to the user query: \n" \
                 "{snippet}\n" \
                 "Now, please reply only yes or no to indicate if the user query should be sent to the search engine.\n" \
                 "Response:"

defalut_prompt_name = \
                 "Federated search retrieves information from a variety of sources via a search application built on top of one or more search engines. A user makes a single query request. The federated search then selects only the search engines that the query should be sent to from a list of search engines, and aggregates the result for presentation of high quality result to the user. The task is called resource selection." \
                 "The following is a search engine with its name and url\n\n" \
                 "Name: {name}\n" \
                 "URL: {url}\n" \
                 "The following is a real user query: \n" \
                 "{query}\n" \
                 "Now, please reply only yes or no to indicate if the query should be sent to the search engine.\n" \
                 "Response:"




class dataset(Dataset):
    def __init__(self, path):
        self.titles = []
        self.abstracts = []
        files = [f for f in os.listdir(path) if f.endswith('.json')]
        for file in files:
            with open(os.path.join(path, file), 'r') as f:
                contents = json.load(f)

            for item in contents:
                if 'description' in item:
                    if 'snippets' in item:
                        title = default_prompt_name_snippet.format(name=item['name'], url=item['url'], query=item['query'], snippet=item['snippet'])
                    else:
                        title = defalut_prompt_name_description.format(name=item['name'], url=item['url'], description=item['description'], query=item['query'])
                else:
                    title = defalut_prompt_name.format(name=item['name'], url=item['url'], query=item['query'])
                abstract = item['output']
                self.titles.append(title)
                self.abstracts.append(abstract)

    def __getitem__(self, index):
        title = self.titles[index]
        abstract = self.abstracts[index]
        return title, abstract

    def __len__(self):
        return len(self.titles)



class DataCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        title_to_abs_inputs = []
        title_to_abs_target = []
        for title, abstract in batch:
            title_to_abs_inputs.append(title)  # Prompt for title_to_abs task.
            title_to_abs_target.append(abstract)

        title_to_abs_inputs = self.tokenizer(title_to_abs_inputs, return_tensors='pt', padding=True, truncation=True)
        title_to_abs_labels = self.tokenizer(title_to_abs_target, return_tensors='pt', padding=True, truncation=True).input_ids

        return title_to_abs_inputs, title_to_abs_labels


class FLANRS(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        outputs = self.model(**inputs, return_dict=True)
        return outputs

    def training_step(self, batch, batch_idx):
        title_to_abs_inputs, title_to_abs_labels = batch

        title_to_abs_loss = self.model(**title_to_abs_inputs,
                                       labels=title_to_abs_labels).loss

        loss = title_to_abs_loss
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return optimizer


if __name__ == "__main__":
    seed_everything(313)

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="t5-large")
    parser.add_argument("--output_path", type=str, default="ckpts/BiTAG")

    # training config
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_nodes", type=int, default=1, help="For multi-node multi-gpu training.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of works for dataloader.")
    parser.add_argument("--max_epochs", type=int, default=10, help="The maximum training epochs.")
    parser.add_argument("--data", type=str, default="/scratch/project/neural_ir/dylan/LLM_FS/fine_tune_model/train_data/", help="where is the training data")
    parser.add_argument("--save_on_steps", type=int, default=2000, help="training steps")
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    wandb.login(key=open("wandb_api", 'r').read().strip())


    wandb.init(project='LLM_FS', entity='shuai-wang2')
    wandb.config.update(args)

    config = T5Config.from_pretrained(args.base_model,
                                      cache_dir="/scratch/project/neural_ir/dylan/LLM_FS/fine_tune_flan/cache",
                                      use_cache=False,
                                      gradient_checkpointing=True,  # trade-off GPU training speed and memory.
                                      )
    model = T5ForConditionalGeneration.from_pretrained(args.base_model, cache_dir="/scratch/project/neural_ir/dylan/LLM_FS/fine_tune_flan/cache", config=config)
    tokenizer = T5Tokenizer.from_pretrained(args.base_model, cache_dir="./scratch/project/neural_ir/dylan/LLM_FS/fine_tune_flan/cache")

    dataset = dataset(args.data)

    wandb_logger = WandbLogger()


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_path,  # Directory to save the checkpoints
        filename='{epoch}-{step}',  # Filename format
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=1,  # Save checkpoint every epoch
    )

    loader = DataLoader(dataset=dataset,
                        batch_size=args.batch_size,
                        drop_last=True,
                        pin_memory=True,
                        shuffle=True,
                        num_workers=args.num_workers,
                        collate_fn=DataCollator(tokenizer))

    trainer = Trainer(max_epochs=args.max_epochs,
                      num_nodes=args.num_nodes,
                      logger=wandb_logger,
                      accelerator="cuda",
                      log_every_n_steps=10,
                      default_root_dir=args.output_path,
                      callbacks=[checkpoint_callback]
                      )
    trainer.fit(FLANRS(model), loader)
