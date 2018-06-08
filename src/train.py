from modules import TextSeg, Trainer

model = TextSeg(lstm_dim=200, score_dim=300, bidir=True, num_layers=2)

trainer = Trainer(model=model,
                  train_dir='../data/wiki_727/train', 
                  val_dir='../data/wiki_50',
                  batch_size=10,
                  lr=1e-3)

trainer.train(num_epochs=30, steps=10)