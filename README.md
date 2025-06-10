git remote add origin https://github.com/zawwanahtue/image-classification

git pull origin main

// enter to project root folder

python -m venv venv

./venv/Script/activate

pip install -r requirements.txt

uvicorn app.main:app --reload
