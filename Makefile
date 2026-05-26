lint:
	ruff check src

format:
	ruff format src

cp2:
	python src/train_cp2.py

api:
	uvicorn src.app:app --reload
