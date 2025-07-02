import os
import time
from datetime import datetime
import pdfplumber
from tqdm import tqdm
import tiktoken
from openai import OpenAI
from openai import APIConnectionError, BadRequestError

# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------
PDF_PATH = r"C:\Users\drodriguez\Downloads\vaia\facturaPDF.pdf"
PROMPT_PATH = r"C:\Users\drodriguez\Desktop\Scripts_py\prompt.txt"
OUTPUT_PATH = "factura_processed.xml"
LOG_PATH = "error.log"

API_KEY = "sk-328e19cd11444733bd4f3ce6a30c3be1"
BASE_URL = "https://api.deepseek.com"

# --------------------------------------------------
# FUNCIONES DE UTILIDAD
# --------------------------------------------------
def log_error(mensaje):
    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(f"[{datetime.now().isoformat()}] {mensaje}\n")

def extract_pdf_text(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"El archivo PDF no existe: {path}")
    
    texto_pdf = ""
    with pdfplumber.open(path) as pdf:
        if len(pdf.pages) == 0:
            raise ValueError("El PDF no contiene páginas.")
        
        for page in tqdm(pdf.pages, desc="📰 Extrayendo texto del PDF"):
            texto = page.extract_text()
            if not texto:
                print(f"⚠️ Página {page.page_number} sin texto")
                continue
            texto_pdf += texto + "\n"
    
    if not texto_pdf.strip():
        raise ValueError("No se extrajo texto útil del PDF.")
    
    return texto_pdf.strip()

# --------------------------------------------------
# 🕒 MEDICIÓN GLOBAL
# --------------------------------------------------
start_global = time.perf_counter()

if not API_KEY or "sk-" not in API_KEY:
    msg = "❌ Clave API no válida o vacía."
    print(msg)
    log_error(msg)
    exit()

# --------------------------------------------------
# 1️⃣ EXTRAER TEXTO DEL PDF
# --------------------------------------------------
try:
    start = time.perf_counter()
    texto_pdf = extract_pdf_text(PDF_PATH)
    duration = time.perf_counter() - start
    print(f"✓ Texto extraído del PDF ({len(texto_pdf)} caracteres) en {duration:.2f}s")
except Exception as e:
    msg = f"❌ Error al leer el PDF: {e}"
    print(msg)
    log_error(msg)
    exit()

# --------------------------------------------------
# 2️⃣ LEER PROMPT BASE
# --------------------------------------------------
try:
    start = time.perf_counter()
    if not os.path.exists(PROMPT_PATH):
        raise FileNotFoundError(f"El archivo de prompt no existe: {PROMPT_PATH}")

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_base = f.read().strip()

    if not prompt_base:
        raise ValueError("El archivo de prompt está vacío.")
    
    duration = time.perf_counter() - start
    print(f"✓ Prompt cargado correctamente en {duration:.2f}s")
except Exception as e:
    msg = f"❌ Error al leer el archivo de prompt: {e}"
    print(msg)
    log_error(msg)
    exit()

# --------------------------------------------------
# 3️⃣ PREPARAR MENSAJE
# --------------------------------------------------
mensaje_usuario = f"{prompt_base}\n\nContenido del PDF:\n{texto_pdf}"

# --------------------------------------------------
# 4️⃣ VALIDAR LÍMITE DE TOKENS
# --------------------------------------------------
try:
    encoding = tiktoken.encoding_for_model("gpt-4")  # DeepSeek usa codificación GPT-4
    system_msg = "Eres un experto en convertir facturas a XML con estructura estricta."
    total_tokens = len(encoding.encode(system_msg)) + len(encoding.encode(mensaje_usuario))

    if total_tokens > 8192:
        msg = f"❌ El mensaje supera el límite de tokens permitido (Usados: {total_tokens}/8192). Reduce el PDF o el prompt."
        print(msg)
        log_error(msg)
        exit()

    print(f"✓ Tokens estimados antes de enviar: {total_tokens}/8192")
except Exception as e:
    msg = f"❌ Error al contar tokens: {e}"
    print(msg)
    log_error(msg)
    exit()

# --------------------------------------------------
# 5️⃣ LLAMADA A LA API
# --------------------------------------------------
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

try:
    start = time.perf_counter()
    print("🤖 Procesando factura con DeepSeek...")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": mensaje_usuario}
        ],
        max_tokens=8192,
        stream=False,
        temperature=0.1,
    )
    duration = time.perf_counter() - start

    if not response or not response.choices or not response.choices[0].message:
        raise ValueError("Respuesta vacía o incompleta de la API.")

    resultado = response.choices[0].message.content
    if not resultado.strip():
        raise ValueError("El contenido de la respuesta está vacío.")

    tokens_usados = getattr(response.usage, 'total_tokens', 'desconocido')
    print(f"✓ Procesamiento completado en {duration:.2f}s (Tokens usados: {tokens_usados})")

except APIConnectionError as e:
    msg = f"❌ Error de conexión con la API: {e}"
    print(msg)
    log_error(msg)
    exit()
except BadRequestError as e:
    msg = f"❌ Error en la solicitud a la API: {e}"
    print(msg)
    log_error(msg)
    exit()
except Exception as e:
    msg = f"❌ Error inesperado durante la llamada a la API: {e}"
    print(msg)
    log_error(msg)
    exit()

# --------------------------------------------------
# 6️⃣ GUARDAR XML
# --------------------------------------------------
try:
    start = time.perf_counter()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(resultado)
    duration = time.perf_counter() - start
    print(f"✅ XML guardado en {OUTPUT_PATH} ({len(resultado)} caracteres) en {duration:.2f}s")
except Exception as e:
    msg = f"❌ Error al guardar el archivo XML: {e}"
    print(msg)
    log_error(msg)
    exit()

# --------------------------------------------------
# 7️⃣ RESUMEN FINAL
# --------------------------------------------------
print("\n--- Vista previa del XML (primeros 300 caracteres) ---")
print(resultado[:300] + "...")

total = time.perf_counter() - start_global
print(f"\n🏁 Proceso completo en {total:.2f} segundos.")