# Detector de rostros en vivo en interfaces de video conferencia 

## Resumen
El programa desarrollado despliega una interfaz que permite capturar videos de rostros en interfaces de videoconferencia.

## Requerimientos
* Python 3.8

## Instalación
Desde la carpeta raiz del proyecto ejecutar:

    pip install -r requirements.txt

## Ejecución
Para abrir el programa se debe ejecutar

    python detector.py

## Interfaz
La interfaz contiene los siguientes botones:
* Botón Iniciar: Comienza la captura de imágenes desde la pantalla definida como
monitor principal e inicia la detección de rostros.
* Botón Detener: Cesa la captura de imágenes y la detección de rostros.
* Spin box: Es una caja de texto numérica que permite ingresar el valor del porcentaje
mínimo de probabilidad para detección.
* Botón Set: Define el porcentaje mínimo de probabilidad para que una detección sea
calificada como rostro.
* Botón Salir: Detiene el programa y cierra la ventana.

En la interfaz se muestra la imagen capturada y las detecciones de rostros encontradas.

## Consideraciones
1. Existen dos modelos de redes neuronales dentro de este repositorio:
    * SSD MobileNet V2 FPNLite 320x320 -> Menor precisión, mayor rapidez.
    * SSD MobileNet V2 FPNLite 640x640 -> Mayor precisión, menor rapidez.

    El uso de estas redes neuronales se puede cambiar en la variable **PATH_TO_SAVED_MODEL** del código **detector.py**

    Por defecto se utiliza la red neuronal SSD MobileNet V2 FPNLite 640x640.

2. El programa genera videos por cada rostro detectado y un video con la imagen completa capturada dentro de la carpeta **RESULTADOS/<time_stamp>**

3. Se puede utilizar GPU en la ejecución del programa, para esto se debe instalar los controladores correspondientes al hardware donde se desenvuelve el programa. La recomendación para la evaluación de la red, no en el entrenamiento, es solo utilizar CPU y desactivar GPU.

4. El uso de GPU se puede habilitar(True) o deshabilitar(False) con la variable **GPU** dentro del código **detector.py**