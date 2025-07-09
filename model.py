import pandas as pd
from bs4 import BeautifulSoup
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np
from sklearn.metrics import classification_report

data_path = "dataset.csv"
text_column_name = "email"
label_column_name = "category"

model_name = "distilbert-base-uncased"
test_size = 0.2
num_labels = 2

#call data
#import pandas as pd
df = pd.read_csv(data_path)

#clean data
#from bs4 import BeautifulSoup
class Cleaner():
  def __init__(self):
    pass
  def put_line_breaks(self,text):
    text = text.replace('</p>','</p>\n')
    return text
  def remove_html_tags(self,text):
    cleantext = BeautifulSoup(text, "lxml").text
    return cleantext
  def clean(self,text):
    text = self.put_line_breaks(text)
    text = self.remove_html_tags(text)
    return text

cleaner = Cleaner()
df['text_cleaned'] = df[text_column_name].apply(cleaner.clean)

#<label encoder>
#from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df[label_column_name].tolist())
df['label'] = le.transform(df[label_column_name].tolist())

#<train/test split>
#from sklearn.model_selection import train_test_split
df_train,df_test = train_test_split(df,test_size=test_size)

#<convert to huggingface dataset aka tensor>
#from datasets import Dataset
train_dataset = Dataset.from_pandas(df_train)
test_dataset = Dataset.from_pandas(df_test)

#<tokenizer>
#from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
def preprocess_function(examples):
    return tokenizer(examples["text_cleaned"], truncation=True)
  
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

#<init model>
#from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

#<train model>
#from transformers import DataCollatorWithPadding
#from transformers import TrainingArguments, Trainer
#import evaluate
#import numpy as np
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_strategy="epoch",
    report_to="none",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics

)
trainer.train()
trainer.save_model('spam_model')

#<evaluate model>
#from sklearn.metrics import classification_report
preds = trainer.predict(tokenized_train)
preds = np.argmax(preds[:3][0],axis=1)
GT = df_train['label'].tolist()
print(classification_report(GT,preds))

preds = trainer.predict(tokenized_test)
preds = np.argmax(preds[:3][0],axis=1)
GT = df_test['label'].tolist()
print(classification_report(GT,preds))