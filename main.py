import cv2
import pytesseract
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

INTERATIVO_ROI = False # True: abre a janela p/ escolher região de interesse

# Definição da região de interesse
if INTERATIVO_ROI:
    frame_demo = cv2.imread("frame_exemplo.jpg")
    (x, y, w, h) = cv2.selectROI("Selecione a região de interesse e aperte ENTER", frame_demo, False, False)
    cv2.destroyAllWindows()
    ROI = (x, y, w, h)
    print("ROI escolhida =", ROI)
else:
    ROI = (792, 505, 530, 278)# (816, 520, 472, 209) # (785, 501, 558, 259) (798, 495, 522, 265) (792, 505, 530, 278) (793, 487, 522, 286)

LIMITE_VAR = 0.1   # 10%
ultimo_valido = None

# Configurações gerais
VIDEO_PATH    ="video.mp4"
INTERVALO_S   = 10  # intervalo de tempo (s)
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
ARQ_TEMP      = "temperatura_glicerina.txt"

DIR_FRAMES, DIR_DEBUGROI, CSV_SAIDA = "frames", "debug_roi", "valores.csv"
SALVAR_FRAMES, SALVAR_DEBUG, MOSTRAR_ROI = True, True, True

#  prepara pastas
if SALVAR_FRAMES: os.makedirs(DIR_FRAMES,   exist_ok=True)
if SALVAR_DEBUG:  os.makedirs(DIR_DEBUGROI, exist_ok=True)

# configura Tesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
OCR_CONFIG = (
    "--oem 1 "      # Tesseract usará apenas o  mecanismo LSTM,baseado em redes neurais recorrentes
    "--psm 7 "                    # Uma unica linha de texto
    "-c tessedit_char_whitelist=0123456789. " # só lê números e ponto e espaço
    "-c classify_bln_numeric_mode=1 " )  # só dicionário numérico, força modo numérico

# abre vídeo
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Não conseguiu abrir o vídeo.")

fps   = cap.get(cv2.CAP_PROP_FPS)
pular = int(fps * INTERVALO_S)

leituras = []  # [(tempo_int, valor|NaN)]
frame_id   = 0
sample_idx = 0    # 0s,1s,2s…

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # salva frame completo
    if SALVAR_FRAMES:
        cv2.imwrite(f"{DIR_FRAMES}/frame_{sample_idx:06d}.jpg", frame)

    # recorta ROI
    x, y, w, h = ROI
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        print(f"Frame {sample_idx}: ROI vazia — verifique coordenadas!")
        break

    # pré‑processamento

    gray   = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    _, thr = cv2.threshold(gray, 60, 255,
                           cv2.THRESH_BINARY_INV)  #cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)  # preenche falhas
    thr = cv2.erode(thr, kernel, iterations=2)  # volta à espessura


    if MOSTRAR_ROI:
        cv2.imshow("ROI", thr)
        if cv2.waitKey(1) & 0xFF == 27: break
    if SALVAR_DEBUG:
        cv2.imwrite(f"{DIR_DEBUGROI}/roi_{sample_idx:06d}.png", thr)

    # OCR (Reconhecimento Óptico de Caracteres)
    texto = pytesseract.image_to_string(thr, config=OCR_CONFIG)\
                       .strip().replace(",", ".")
    try:
        valor = round(float(texto), 2)
    except ValueError:
        valor = "NaN"
    print(f"lido ---> {texto}")     # verificação do valor lido antes do filtro
    try:
        valor_num = round(float(texto), 2)
    except ValueError:
        valor_num = math.nan  # OCR falhou

    # Filtro para valores fora da %
    if not math.isnan(valor_num) and ultimo_valido is not None:
        variacao = abs(valor_num - ultimo_valido) / ultimo_valido
        if variacao > LIMITE_VAR:
            valor_num = math.nan  # descarta como fora de faixa
    if not math.isnan(valor_num): # atualiza referência se o valor for válido
        ultimo_valido = valor_num
    valor = valor_num if not math.isnan(valor_num) else "NaN"

    tempo_s = sample_idx * INTERVALO_S
    leituras.append((tempo_s, valor))
    print(f"{tempo_s:4d} s    {valor} cP")

    # avança
    sample_idx += 1
    frame_id  += pular
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

cap.release()
cv2.destroyAllWindows()

# Grava a viscosidade em CSV
with open(CSV_SAIDA, "w", newline="") as f:
    csv.writer(f, delimiter=";").writerows(
        [("tempo_s", "viscosidade_cP"), *leituras]
    )
print(f"\n{len(leituras)} linhas salvas em '{CSV_SAIDA}'")

# Leitura dos arquivos para plotar

# arquivos de entrada
CSV_VISC = "valores.csv"
ARQ_TEMP = "temperatura_glicerina.txt"

# lê viscosidade
df_visc = pd.read_csv(CSV_VISC, sep=";")
df_visc["viscosidade_cP"] = pd.to_numeric(df_visc["viscosidade_cP"], errors="coerce")
df_visc["tempo_s"]        = pd.to_numeric(df_visc["tempo_s"], errors="coerce").astype(int)

# lê temperatura
df_temp_raw = pd.read_csv(
    ARQ_TEMP,
    sep=r"[ \t]+",    # separador: espaço(s) ou tab
    decimal=",",      # vírgula decimal
    engine="python",
    header=0
)
df_temp = df_temp_raw.iloc[:, :2].copy()
df_temp.columns = ["tempo_s", "temperatura_C"]
df_temp["tempo_s"]        = pd.to_numeric(df_temp["tempo_s"], errors="coerce").astype(int)
df_temp["temperatura_C"]  = pd.to_numeric(df_temp["temperatura_C"], errors="coerce")

# merge (mescla) pelo tempo mais próximo (±1s)
df_merged = pd.merge_asof(
    df_visc.sort_values("tempo_s"),
    df_temp.sort_values("tempo_s"),
    on="tempo_s",
    direction="nearest",
    tolerance=1
).dropna(subset=["viscosidade_cP", "temperatura_C"])

# salva combinado viscosidade e temperatura
df_merged.to_csv("temp_vs_visc.csv", index=False, sep=";")

# Cálculo da viscosidade teórica

# df_merged["viscosidade_teorica_cP"] = 11230 * np.exp(-0.0905 * df_merged["temperatura_C"])
x = df_merged["temperatura_C"]
df_merged["viscosidade_teorica_cP"] = (
    12059 + (-1283) * x + 60.3 * x**2 + (-1.51) * x**3 + 0.0205 * x**4 + (-1.43e-4) * x**5 + 3.98e-7 * x**6)
df_merged.to_csv("valores_viscosidade_teorica.csv", index=False, sep=";")

# Adimensionalização da viscosidade

# Escolhe a viscosidade teórica na menor temperatura como referência
mu0_teo = df_merged["viscosidade_teorica_cP"].iloc[0]
mu0_exp = df_merged["viscosidade_cP"].iloc[0]

# Adimensionaliza
df_merged["viscosidade_exp_adim"] = df_merged["viscosidade_cP"] / mu0_exp
df_merged["viscosidade_teo_adim"]= df_merged["viscosidade_teorica_cP"] / mu0_teo

# Salva CSV com viscosidade adimensionalizada
df_merged.to_csv("valores_viscosidade_adimensional.csv", index=False, sep=";")

# Gráfico adimensional
plt.figure(figsize=(10, 5))

plt.plot(df_merged["temperatura_C"],
         df_merged["viscosidade_exp_adim"],
         "ro-", label="Experimental (adim)")

plt.plot(df_merged["temperatura_C"],
         df_merged["viscosidade_teo_adim"],
         "bo-", label="Teórica (adim)")

plt.xlabel("Temperatura (°C)")
plt.ylabel("Viscosidade adimensionalizada (μ/μ₀)")
plt.title("Temperatura vs. Viscosidade Adimensionalizada")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#gráfico sem adimensionalizar

plt.figure(figsize=(10, 5))

plt.scatter(df_merged["temperatura_C"],    # dados experimentais
            df_merged["viscosidade_cP"],
            c="r",linewidth=2, label="Experimental")

plt.scatter(df_merged["temperatura_C"],    # curva teórica
         df_merged["viscosidade_teorica_cP"],
         c="blue", linewidth=2, label="Teórica")

plt.xlabel("Temperatura (°C)")
plt.ylabel("Viscosidade (cP)")
plt.title("Temperatura vs. Viscosidade")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
