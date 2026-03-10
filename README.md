# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
To identify and classify named entities in a given text. Named Entity Recognition (NER) is a subtask of information extraction that seeks to locate and classify named entity mentions in unstructured text into predefined categories such as person names, organizations, geographical locations, time expressions, etc.

<img width="527" height="366" alt="image" src="https://github.com/user-attachments/assets/8a3d4a3b-e47a-4220-8910-38ae341db1a5" />


## DESIGN STEPS
### STEP 1: 
Data Preparation and Encoding: Load data, create word/tag mappings, encode sentences, and pad sequences to a fixed length.

### STEP 2: 
Dataset and DataLoader Setup: Implement a PyTorch Dataset for encoded data and DataLoaders for batch processing.

### STEP 3: 
Bi-LSTM Model Architecture: Define a Bi-LSTM network with embedding, dropout, LSTM, and a linear output layer.

### STEP 4: 
Model Initialization and Optimization: Instantiate the model, define CrossEntropyLoss, and set up the Adam optimizer.


### STEP 5: 
Training Loop Implementation: Create a training function to run epochs, perform backpropagation, and track loss.


### STEP 6: 
Evaluation and Prediction: Develop functions to evaluate model performance and make predictions on new input sentences.




## PROGRAM

### Name:Eesha Ranka

### Register Number:212224240040

```python
class BiLSTMTagger(nn.Module):
  def __init__(self,vocab_size,target_size,embedding_dim=50,hidden_dim=100):
    super(BiLSTMTagger,self).__init__()
    self.embedding=nn.Embedding(vocab_size,embedding_dim)
    self.dropout=nn.Dropout(0.1);
    self.lstm=nn.LSTM(embedding_dim,hidden_dim,batch_first=True,bidirectional=True)
    self.fc=nn.Linear(hidden_dim*2,target_size)


  def forward(self, x):
    x=self.embedding(x)
    x=self.dropout(x)
    x,_=self.lstm(x)
    return self.fc(x)
        


model =BiLSTMTagger(len(word2idx)+1,len(tag2idx)).to(device)
loss_fn =nn.CrossEntropyLoss()
optimizer =torch.optim.Adam(model.parameters(),lr=0.001) 


# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses,val_losses=[],[]
    for epoch in range(epochs):
      model.train()
      total_loss=0;
      for batch in train_loader:
        input_ids=batch["input_ids"].to(device)
        labels=batch["labels"].to(device)
        optimizer.zero_grad()
        outputs=model(input_ids)
        loss=loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
      train_losses.append(total_loss)

      model.eval()
      val_loss=0
      with torch.no_grad():
        for batch in test_loader:
          input_ids=batch["input_ids"].to(device)
          labels=batch["labels"].to(device)
          outputs=model(input_ids)
          loss=loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))
          val_loss+=loss.item()
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}: Train_loss={total_loss:.4f},val loss={val_loss:.4f}")
    return train_losses, val_losses


```

### OUTPUT

## Loss Vs Epoch Plot

<img width="571" height="455" alt="download" src="https://github.com/user-attachments/assets/eb291096-9301-4121-879c-4b9c29912763" />

### Sample Text Prediction\

<img width="915" height="580" alt="image" src="https://github.com/user-attachments/assets/5430cccf-c8a7-4df4-865a-1f78e87d0e85" />

<img width="612" height="623" alt="image" src="https://github.com/user-attachments/assets/7b8ba925-0f31-4e98-890f-27eaf1e7b528" />


## RESULT
Thus the model has been trained.
