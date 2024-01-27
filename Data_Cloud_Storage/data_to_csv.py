import subprocess
from google.cloud import storage
from google.auth import default

# Ver Buckets creados en la cuenta de Google actual
def list_buckets():
    try:
        # Crea un cliente de Google Cloud Storage
        storage_client = storage.Client()
        buckets = storage_client.list_buckets()

        # Lista los buckets disponibles
        print("Buckets disponibles:")
        for bucket in buckets:
            print(bucket.name)

    except Exception as e:
        print(f"Ocurrió un error: {e}")

# Obtener información de la cuenta de Google actual
def get_gcloud_account_info():
    try:
        # Comando para listar todas las cuentas
        all_accounts = subprocess.check_output("gcloud auth list", shell=True).decode('utf-8')
        print("Todas las cuentas:")
        print(all_accounts)

        # Comando para mostrar la cuenta actual
        current_account = subprocess.check_output("gcloud config get-value account", shell=True).decode('utf-8').strip()
        print("Cuenta actual:")
        print(current_account)

    except subprocess.CalledProcessError as e:
        print(f"Error al obtener la información de la cuenta: {e}")

# Verificar las credenciales de la cuenta de Google actual
def check_credentials():
    try:
        # Obtener credenciales actuales
        credentials, _ = default()
        if credentials is not None:
            print("Información de credenciales actuales:")
            print(credentials)
        else:
            print("No se encontraron credenciales de cuenta de servicio.")

    except Exception as e:
        print(f"Error al verificar las credenciales: {e}")

# Ejecutar las funciones
get_gcloud_account_info()
check_credentials()
list_buckets()

# Codigo para generar el CSV con las imagenes de Cloud Storage
from google.cloud import storage
import csv

# Configurar cliente de Google Cloud Storage
client = storage.Client()
bucket = client.get_bucket('brain_check_bucket') #Acá va el nombre del bucket

# Lista los archivos en el bucket
blobs = bucket.list_blobs()

# Prepara los datos para el CSV
data = []

for blob in blobs:
    if 'jpg' in blob.name:  # Filtra solo imágenes JPEG
        uri = f"gs://{bucket.name}/{blob.name}"
        label = blob.name.split('/')[0]  # Asume que la etiqueta es el primer directorio
        data.append([uri, label])

# Escribe los datos en un archivo CSV
with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image URI', 'Label'])  # Encabezado del CSV
    writer.writerows(data)

print("CSV generado con éxito.")
