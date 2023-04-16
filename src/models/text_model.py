# coding=utf-8
from transformers import XLMRobertaTokenizer, XLMRobertaModel, AdamW, DataCollatorWithPadding
import pickle
import torch
import pytorch_lightning as pl
import argparse
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import accuracy_score


TOKENIZER = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')


class TitleModel(torch.nn.Module):
    def __init__(self, dropout = 0.0):
        super().__init__()
        self.bert = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.fc = torch.nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.proj = torch.nn.Linear(self.bert.config.hidden_size, 3)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.head = torch.nn.Linear(self.bert.config.hidden_size, 3)
        self.activation = torch.nn.ReLU()

    def forward(self, input_ids):
        attention_mask = input_ids != self.pad_token_id

        x = self.bert(input_ids, attention_mask=attention_mask)[0][:, 0, :]
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.proj(x)

        return x


class TitleDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data,
            tokenizer,
            zone,
            max_len
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.zone = zone

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        input_ids, label = item[self.zone], item['class']
        input_ids = torch.tensor(input_ids)
        input_ids, __, __ = self.tokenizer.truncate_sequences(
            input_ids, num_tokens_to_remove=len(input_ids) - self.max_len
        )

        return {'input_ids': input_ids, 'labels': label}


class LitModel(pl.LightningModule):
    def __init__(self, lr, freeze_embeddings, wd, epochs, dropout):
        super().__init__()
        self.save_hyperparameters()

        self.model = TitleModel(dropout=dropout)
        if self.hparams.freeze_embeddings:
            for parameter in self.model.bert.embeddings.parameters():
                parameter.requires_grad = False

        self.tokenizer = TOKENIZER
        self.model.pad_token_id = self.tokenizer.pad_token_id
        self.criterion = torch.nn.CrossEntropyLoss()

    def accuracy__(self, scores, labels):
        return accuracy_score(labels, scores.argmax(axis=1))

    def forward(self, input_ids):
        return self.model(input_ids)

    def training_step(self, batch, batch_index):
        input_ids, labels = batch['input_ids'], batch['labels']
        scores = self(input_ids)
        loss = self.criterion(scores, labels)
        return {
            'loss': loss,
            'labels': labels,
            'scores': scores
        }

    def training_epoch_end(self, training_step_outputs):
        loss, labels, scores = [], [], []
        for output in training_step_outputs:
            loss.append(output['loss'])
            labels.append(output['labels'])
            scores.append(output['scores'])

        loss = torch.mean(torch.stack(loss))
        labels = torch.cat(labels).cpu().detach().numpy()
        scores = torch.cat(scores).cpu().detach().numpy()

        self.log('train_loss', loss)
        self.log('train_accuracy', self.accuracy__(scores, labels))

    def validation_step(self, batch, batch_index):
        input_ids, labels = batch['input_ids'], batch['labels']
        scores = self(input_ids)

        return {'scores': scores, 'labels': labels}

    def validation_epoch_end(self, validation_step_outputs):
        scores, labels = [], []
        for output in validation_step_outputs:
            scores.append(output['scores'])
            labels.append(output['labels'])

        scores = torch.cat(scores)
        labels = torch.cat(labels)

        self.log('val_loss', self.criterion(scores, labels))
        self.log('val_accuracy', self.accuracy__(scores.cpu().numpy(), labels.cpu().numpy()))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.lr,
                                                        steps_per_epoch=10, epochs=self.hparams.epochs)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }


class LitDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, test_dataset,
                 train_bs, test_bs, max_len, zone):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = TOKENIZER

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            with open(self.hparams.train_dataset, 'rb') as f:
                train = pickle.load(f)

            with open(self.hparams.test_dataset, 'rb') as f:
                test = pickle.load(f)

            self.train_dataset, self.val_dataset = (TitleDataset(
                data=data,
                tokenizer=self.tokenizer,
                max_len=self.hparams.max_len,
                zone=self.hparams.zone
            ) for data in (train, test))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, collate_fn=DataCollatorWithPadding(tokenizer=self.tokenizer),
            num_workers=8,
            batch_size=self.hparams.train_bs,
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, collate_fn=DataCollatorWithPadding(tokenizer=self.tokenizer),
            num_workers=8,
            batch_size=self.hparams.test_bs,
            shuffle=False
        )


def main(args):
    model = LitModel(
        args.lr, args.freeze_embeddings,
        wd=args.wd,
        epochs=args.epochs,
        dropout=args.dropout
    )
    data_module = LitDataModule(args.train_dataset, args.test_dataset,
                                args.train_bs, args.test_bs, args.max_len, args.zone)
    wandb_logger = WandbLogger(project="scp_classes")
    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=args.gpus,
        strategy='ddp',
        max_epochs=args.epochs,
        num_sanity_val_steps=0
    )
    trainer.fit(model, data_module)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zone', type=str, default='title')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--train_dataset', type=str, default='data/titles/train.pkl')
    parser.add_argument('--test_dataset', type=str, default='data/titles/test.pkl')
    parser.add_argument('--train_bs', type=int, default=16)
    parser.add_argument('--test_bs', type=int, default=16)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--freeze_embeddings', action='store_true', dest='freeze_embeddings')
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.0)

    return parser.parse_args()


if __name__ == '__main__':
    main(get_args())
