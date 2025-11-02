FROM python:3.12-slim

WORKDIR /app

ENV PS1="🐳 \u@\h:\w\$ "

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mlops_hp_2.py .

CMD ["python", "mlops_hp_2.py"]