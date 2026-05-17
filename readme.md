# Instrucciones de Ejecución

## 1. Crear entorno virtual

Mac/Linux:

python3 -m venv venv

Windows:

python -m venv venv


--------------------------------------------------

# 2. Activar entorno virtual

Mac/Linux:

source venv/bin/activate

Windows:

venv\Scripts\activate

Cuando esté activado aparecerá algo así:

(venv)


--------------------------------------------------

# 3. Instalar dependencias

pip install -r requirements.txt

--------------------------------------------------

# 4. Entrenar el modelo Transformer

Ejecutar:

python src/train.py

Durante el entrenamiento el sistema:

- Descargará el modelo FLAN-T5
- Tokenizará el dataset
- Entrenará el modelo
- Guardará el modelo entrenado


--------------------------------------------------

# 5. Ejecutar demo

Después del entrenamiento ejecutar:

python src/demo.py


--------------------------------------------------

# Resultado esperado

=========== DEMO TRANSFORMER ===========

TRADUCCIÓN
Entrada: Translate Spanish to English: Estoy estudiando ingeniería de software.
Salida: I am studying software engineering.

-----------------------------------

CLASIFICACIÓN
Entrada: Classify the sentiment as positivo, negativo or neutral: El servicio fue horrible.
Salida: negativo