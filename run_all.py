import subprocess
import os
import signal
import time
import sys
import atexit

# Global process handle for the API (used to terminate on Ctrl-C)
API_PROCESS = None

VENV_DIR = ".venv"
REQUIREMENTS_FILE = "requirements.txt"


def manage_venv():
    if not os.path.isdir(VENV_DIR):
        print(f"--- 1. Entorno virtual '{VENV_DIR}' no encontrado. Creando... ---")
        try:
            subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
            print("✅ Entorno virtual creado.")
        except Exception as e:
            print(f"❌ ERROR al crear el venv: {e}")
            sys.exit(1)

    if sys.platform.startswith('win'):
        python_executable = os.path.join(VENV_DIR, 'Scripts', 'python.exe')
        pip_executable = os.path.join(VENV_DIR, 'Scripts', 'pip.exe')
    else:
        python_executable = os.path.join(VENV_DIR, 'bin', 'python')
        pip_executable = os.path.join(VENV_DIR, 'bin', 'pip')

    print(f"--- 2. Instalando/Verificando dependencias desde {REQUIREMENTS_FILE} ---")
    try:
        subprocess.run([pip_executable, "install", "-r", REQUIREMENTS_FILE], check=True)
        print("✅ Dependencias instaladas/verificadas.")
        return python_executable
    except subprocess.CalledProcessError:
        print("❌ ERROR al instalar dependencias. Revise requirements.txt.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ ERROR: No se encontró el ejecutable 'pip' en {VENV_DIR}. Inténtelo de nuevo.")
        sys.exit(1)


def run_pipeline(python_executable):
    TRAIN_COMMAND = [python_executable, "model_train.py"]
    API_COMMAND = [python_executable, "-m", "uvicorn", "predict_api:app", "--reload"]
    DASHBOARD_COMMAND = [python_executable, "-m", "streamlit", "run", "streamlit_dashboard.py"]

    print("\n--- 3. EJECUTANDO ENTRENAMIENTO Y PREPROCESAMIENTO ---")
    try:
        subprocess.run(TRAIN_COMMAND, check=True)
        print("✅ Entrenamiento y artefactos generados exitosamente.")
    except subprocess.CalledProcessError:
        print("❌ ERROR: El entrenamiento falló. Revise model_train.py.")
        sys.exit(1)

    print("\n--- 4. INICIANDO LA API DE FASTAPI EN SEGUNDO PLANO ---")
    api_process = None
    try:
        api_process = subprocess.Popen(API_COMMAND, preexec_fn=os.setsid,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"API iniciada con PID: {api_process.pid}. Accede a http://127.0.0.1:8080/docs")
        print("Esperando 5 segundos para que la API se inicialice...")
        time.sleep(5)

        # Guardar el handle globalmente y registrar handlers de señal
        global API_PROCESS
        API_PROCESS = api_process

        def _terminate_api(signum, frame):
            print("\n--- Señal recibida: terminando la API... ---")
            try:
                if API_PROCESS is not None:
                    os.killpg(os.getpgid(API_PROCESS.pid), signal.SIGTERM)
                    print(f"API (PID: {API_PROCESS.pid}) terminada mediante señal.")
            except Exception as _e:
                print(f"Error al intentar terminar la API: {_e}")
            sys.exit(0)

        signal.signal(signal.SIGINT, _terminate_api)
        signal.signal(signal.SIGTERM, _terminate_api)

    except Exception as e:
        print(f"❌ ERROR: No se pudo iniciar la API. Error: {e}")

    print("\n--- 5. INICIANDO EL DASHBOARD DE STREAMLIT ---")
    print("El Dashboard se abrirá en su navegador. Cierre la pestaña o presione Ctrl+C en la terminal para terminar.")
    try:
        subprocess.run(DASHBOARD_COMMAND, check=True)
    except Exception as e:
        print(f"❌ ERROR al ejecutar Streamlit: {e}")

    if api_process:
        print("\n--- 6. DETENIENDO LA API DE FASTAPI ---")
        try:
            os.killpg(os.getpgid(api_process.pid), signal.SIGTERM)
            print("✅ API detenida exitosamente.")
        except Exception as e:
            print(f"Advertencia: No se pudo detener la API automáticamente. Detenga el proceso (PID: {api_process.pid}) manualmente. Error: {e}")

    # Registrar limpieza automática (atexit)
    def _atexit_cleanup():
        try:
            if API_PROCESS is not None and API_PROCESS.poll() is None:
                os.killpg(os.getpgid(API_PROCESS.pid), signal.SIGTERM)
                print(f"API (PID: {API_PROCESS.pid}) detenida en atexit.")
        except Exception:
            pass
    atexit.register(_atexit_cleanup)

    print("\n--- 🏁 PIPELINE COMPLETO TERMINADO. ---")


if __name__ == "__main__":
    if os.environ.get('VIRTUAL_ENV') is None:
        python_exec = manage_venv()
    else:
        print("--- Usando el entorno virtual activo. ---")
        python_exec = sys.executable
    run_pipeline(python_exec)
