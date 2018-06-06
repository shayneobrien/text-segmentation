from modules import TextSeg, Trainer

model = TextSeg(lstm_dim=300, score_dim=300, bidir=True, num_layers=2)
trainer = Trainer(model, '../data/wiki_727/train', batch_size=20)
trainer.train(30)
