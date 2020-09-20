help:
	@echo ""
	@echo "venv: "
	@echo "    Create a Python 3.7.6 pyenv virtual environment."
	@echo ""
	@echo "init:"
	@echo "    Install requirements and fetch data."
	@echo ""

venv:
	pyenv virtualenv 3.7.6 titanic
	@echo ""
	@echo "Virtual environment titanic created."
	@echo ""
	@echo "Activate virtual environment with: "
	@echo "    pyenv activate titanic"
	@echo ""

init:
	pip install -r requirements.txt
	mkdir data 
	cd data && kaggle competitions download -c titanic
	cd data && unzip titanic.zip
	cd data && rm titanic.zip
	@echo "Packages installed and data fetched."