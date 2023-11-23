# financial_sentiment_finetuning

This is to demonstrate how to fine tune pretrained LLMs to our own dataset and make predcitions. 

1. Set up virtual env from root directory using python -m venv .venv. Then activate venv by running the activate.bat file in .venv/Scripts.
2. Run pip install -r requirements.txt and make sure all dependencies are installed in .venv/Scripts folder. 
3. Make sure the data from https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news (all_data.csv) is downloaded and provide the full path in configs.py without fail. Or the model training will fail. You can also place similar sentiment datasets and train on it but make sure to change the name of input and output columns in configs.py accordingly.
4. Run main.py using "python main.py" from a terminal (root directory of repo)
