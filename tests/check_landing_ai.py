import requests
import sys

def probar_conexion():
    """
    Prueba la conexión contra el endpoint 'Agentic Document Extraction'
    usando una API key pegada directamente.
    """
    


    # 2. Verificación (no edites esto)
    if MI_API_KEY == "pega_tu_api_key_aqui" or MI_API_KEY == "":
        print("\n-----------------------------------------------------")
        print("❌ ERROR: Edita este archivo (test_mi_conexion.py) primero.")
        print("Pega tu API key real en la variable 'MI_API_KEY'.")
        print("-----------------------------------------------------")
        sys.exit(1) # Salir

    # 3. Endpoint y Headers
    # (Endpoint del servicio 'Agentic Document Extraction' 
    #  que coincide con la documentación de tu "Explore Plan")
    
    URL_DEL_SERVICIO = "https://api.va.landing.ai/v1/ade/parse"
    
    headers = {
        "Authorization": f"Bearer {MI_API_KEY}"
    }

    print(f"Intentando conectar a: {URL_DEL_SERVICIO} ...")

    try:
        # 4. Hacemos la llamada de prueba
        # Hacemos un POST vacío a propósito.
        # Si la KEY es VÁLIDA, el servidor nos reconocerá y
        # responderá 400 (Bad Request) o 422 (Unprocessable Entity).
        # Si la KEY es INVÁLIDA, responderá 401 (Unauthorized).
        
        response = requests.post(URL_DEL_SERVICIO, headers=headers)

        # 5. Interpretar la respuesta
        
        if response.status_code == 401:
            print("\n-----------------------------------------------------")
            print("❌ CONEXIÓN FALLIDA: 401 Unauthorized")
            print("La API key que pegaste en el script es INCORRECTA,")
            print("ha expirado o no pertenece a este servicio.")
            print("-----------------------------------------------------")
            
        elif response.status_code in [400, 422]:
            print("\n-----------------------------------------------------")
            print(f"✅ ¡CONEXIÓN EXITOSA! (Recibido: {response.status_code})")
            print("El servidor respondió que la solicitud no es válida (lo cual")
            print("es esperado), pero esto confirma que")
            print("tu API key fue ACEPTADA.")
            print("-----------------------------------------------------")
            
        else:
            print(f"\nRespuesta inesperada del servidor: {response.status_code}")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"\nError de conexión de red (¿Estás conectado a internet?): {e}")

if __name__ == "__main__":
    probar_conexion()