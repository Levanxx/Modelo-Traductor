from predict import predict

print("\n=========== DEMO TRANSFORMER ===========\n")

texto1 = "Translate Spanish to English: Estoy estudiando ingeniería de software."
resultado1 = predict(texto1)

print("TRADUCCIÓN")
print("Entrada:", texto1)
print("Salida:", resultado1)

print("\n-----------------------------------\n")

texto2 = "Classify the sentiment as positivo, negativo or neutral: El servicio fue horrible."
resultado2 = predict(texto2)

print("CLASIFICACIÓN")
print("Entrada:", texto2)
print("Salida:", resultado2)

print("\n=======================================\n")