FROM python:3.10-slim

# Instala dependencias del sistema
RUN apt-get update && apt-get install -y build-essential

# Crea y entra en el directorio de la app
WORKDIR /app

# Copia todos los archivos del proyecto
COPY . .

# Instala dependencias Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Exponer puerto de Streamlit
EXPOSE 8501

# Comando para ejecutar la app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
