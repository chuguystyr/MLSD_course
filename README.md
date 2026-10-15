## Запуск

1. Завантажити набір даних Bank Marketing  і покласти файл bank-additional-full.csv у папку data/.  
2. Встановити залежності: pip install -r requirements.txt.  
3. Навчити модель: python train.py.  
4. Отримати прогнози для демонстраційних записів: python predict.py new_records.csv.
5. Запустити вебсервіс: uvicorn app:app --reload
6. Відкрити http://127.0.0.1:8000/docs і виконати POST /predict.
