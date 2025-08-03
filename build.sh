#!/bin/bash

# Nombre del archivo fuente
SOURCE="terrain_glow.cpp"
OUTPUT="terrain_glow"

# Comando de compilación
echo "🔧 Compilando $SOURCE..."
g++ "$SOURCE" -o "$OUTPUT" $(pkg-config --cflags --libs opencv4)

# Verificar si la compilación fue exitosa
if [ $? -eq 0 ]; then
    echo "✅ Compilación completada. Ejecuta con: ./terrain_glow"
else
    echo "❌ Error en la compilación"
fi

