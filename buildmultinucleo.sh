#!/bin/bash

# Nombre del archivo fuente
SOURCE="multinucleo.cpp"
OUTPUT="multinucleo"

# Comando de compilación con soporte para OpenMP
echo "🔧 Compilando $SOURCE con OpenMP..."
g++ -fopenmp -O2 -std=c++17 "$SOURCE" -o "$OUTPUT" $(pkg-config --cflags --libs opencv4)

# Verificar si la compilación fue exitosa
if [ $? -eq 0 ]; then
    echo "✅ Compilación completada. Ejecuta con: ./$OUTPUT"
else
    echo "❌ Error en la compilación"
fi

