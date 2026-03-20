.DEFAULT_GOAL := run

run:
	streamlit run app.py

install:
	pip install -r requirements.txt
