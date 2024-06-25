install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

data:
	mkdir data &&\
	cd data &&\
	kaggle competitions download -c mlg-ulb/creditcardfraud &&\
	unzip mlg-ulb/creditcardfraud.zip &&\
	rm mlg-ulb/creditcardfraud.zip
	
run:
	python main.py

format:
	black *.py

submit:
	kaggle competitions submit -c llm-detect-ai-generated-text -f ./data/submission.csv -m "Message"

score:
	kaggle competitions submissions -c llm-detect-ai-generated-text

lint:
	pylint --disable=R,C main.py

clear:
	rm -r data
	
clear-env:
	rm -r .venv

clear-checkpoint:
	rm -r checkpoint
