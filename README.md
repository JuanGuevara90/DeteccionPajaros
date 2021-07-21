conda activate test
conda activate test

# DeteccionPajaros

Requisitos para ejecutar previamente una vez construido el ambiente

pip install firebase-admin
pip install torch
pip install torchvision
pip install pyserial

#Ingreso a Archiconda  y al ambiente

conda activate test

#Ruta Proyecto

/home/jetson/Documents/VisionArtificialPajaros

#Ruta de carpeta ejecución
/home/jetson/Documents/VisionArtificialPajaros/src

#Ngrok Activar App Web
/home/jetson/Downloads
./ngrok http 8000

#deteccion personas y mas. " Dentro de la carpeta /home/jetson/Documents/VisionArtificialPajaros/src ejecutar el siguiente comando:
python appGeneral.py mb1-ssd models/mobilenet-v1-ssd-mp-0_675.pth models/voc-model-labels.txt 0.8

#Deteccion pajaros " Dentro de la carpeta /home/jetson/Documents/VisionArtificialPajaros/src ejecutar el siguiente comando:
python appPajaro.py mb1-ssd models/bird/mb1-ssd-bird.pth models/bird/labels.txt 0.5
