.PHONY: install data run test clean

install:
	pip install -r requirements.txt

data:
	python download_data.py

run: data
	jupyter nbconvert --to notebook --execute preprocessing.ipynb --output preprocessing_output.ipynb

test:
	python -m pytest tests/ -v

clean:
	rm -f preprocessing_output.ipynb
	rm -rf __pycache__ .pytest_cache
