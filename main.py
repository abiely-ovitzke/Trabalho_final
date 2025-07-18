import cv2
import pytesseract
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import math

INTERACTIVE_ROI = False  # True: open window to select Region of Interest (ROI)

# Define the Region of Interest (ROI)
if INTERACTIVE_ROI:
    frame_demo = cv2.imread("example_frame.jpg")
    (x, y, w, h) = cv2.selectROI("Select the region of interest and press ENTER", frame_demo, False, False)
    cv2.destroyAllWindows()
    ROI = (x, y, w, h)
    print("Selected ROI =", ROI)
else:
    ROI = (792, 505, 530, 278)

VARIATION_LIMIT = 0.1   # 10%
last_valid = None

# General configuration
VIDEO_PATH = "video.mp4"
INTERVAL_S = 10  # time interval (seconds)
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TEMP_FILE = "glycerin_temperature.txt"

FRAME_DIR, DEBUG_ROI_DIR, OUTPUT_CSV = "frames", "debug_roi", "values.csv"
SAVE_FRAMES, SAVE_DEBUG, SHOW_ROI = True, True, True

# Prepare folders
if SAVE_FRAMES: os.makedirs(FRAME_DIR, exist_ok=True)
if SAVE_DEBUG: os.makedirs(DEBUG_ROI_DIR, exist_ok=True)

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
OCR_CONFIG = (
    "--oem 1 "  # Tesseract will use only the LSTM-based engine (RNN)
    "--psm 7 "  # Single line of text
    "-c tessedit_char_whitelist=0123456789. "  # Only read digits, period and space
    "-c classify_bln_numeric_mode=1 "  # Numeric dictionary only, enforce numeric mode
)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Failed to open video.")

fps = cap.get(cv2.CAP_PROP_FPS)
skip = int(fps * INTERVAL_S)

readings = []  # [(time_int, value|NaN)]
frame_id = 0
sample_idx = 0  # 0s,1s,2s…

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Save full frame
    if SAVE_FRAMES:
        cv2.imwrite(f"{FRAME_DIR}/frame_{sample_idx:06d}.jpg", frame)

    # Crop ROI
    x, y, w, h = ROI
    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        print(f"Frame {sample_idx}: Empty ROI — check coordinates!")
        break

    # Pre-processing

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    _, thr = cv2.threshold(gray, 60, 255,
                           cv2.THRESH_BINARY_INV)  #cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)  # fill gaps
    thr = cv2.erode(thr, kernel, iterations=2)  # restore thickness

    if SHOW_ROI:
        cv2.imshow("ROI", thr)
        if cv2.waitKey(1) & 0xFF == 27: break
    if SAVE_DEBUG:
        cv2.imwrite(f"{DEBUG_ROI_DIR}/roi_{sample_idx:06d}.png", thr)

    # OCR (Optical Character Recognition)
    text = pytesseract.image_to_string(thr, config=OCR_CONFIG)\
                       .strip().replace(",", ".")
    try:
        value = round(float(text), 2)
    except ValueError:
        value = "NaN"
    print(f"read ---> {text}")  # Check value read before filtering
    try:
        value_num = round(float(text), 2)
    except ValueError:
        value_num = math.nan  # OCR failed

    # Filter out-of-range values
    if not math.isnan(value_num) and last_valid is not None:
        variation = abs(value_num - last_valid) / last_valid
        if variation > VARIATION_LIMIT:
            value_num = math.nan  # discard as out of range
    if not math.isnan(value_num):  # update reference if value is valid
        last_valid = value_num
    value = value_num if not math.isnan(value_num) else "NaN"

    time_s = sample_idx * INTERVAL_S
    readings.append((time_s, value))
    print(f"{time_s:4d} s    {value} cP")

    # Advance
    sample_idx += 1
    frame_id += skip
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

cap.release()
cv2.destroyAllWindows()

# Save viscosity data to CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    csv.writer(f, delimiter=";").writerows(
        [("time_s", "viscosity_cP"), *readings]
    )
print(f"\n{len(readings)} rows saved to '{OUTPUT_CSV}'")

# Read files for plotting

# Input files
CSV_VISC = "values.csv"
TEMP_FILE = "glycerin_temperature.txt"

# Read viscosity
df_visc = pd.read_csv(CSV_VISC, sep=";")
df_visc["viscosity_cP"] = pd.to_numeric(df_visc["viscosity_cP"], errors="coerce")
df_visc["time_s"] = pd.to_numeric(df_visc["time_s"], errors="coerce").astype(int)

# Read temperature
df_temp_raw = pd.read_csv(
    TEMP_FILE,
    sep=r"[ \t]+",  # separator: space(s) or tab
    decimal=",",  # comma as decimal point
    engine="python",
    header=0
)
df_temp = df_temp_raw.iloc[:, :2].copy()
df_temp.columns = ["time_s", "temperature_C"]
df_temp["time_s"] = pd.to_numeric(df_temp["time_s"], errors="coerce").astype(int)
df_temp["temperature_C"] = pd.to_numeric(df_temp["temperature_C"], errors="coerce")

# Merge by nearest time (±1s)
df_merged = pd.merge_asof(
    df_visc.sort_values("time_s"),
    df_temp.sort_values("time_s"),
    on="time_s",
    direction="nearest",
    tolerance=1
).dropna(subset=["viscosity_cP", "temperature_C"])

# Save combined viscosity and temperature
df_merged.to_csv("time_vs_viscosity.csv", index=False, sep=";")

# Theoretical viscosity calculation

# df_merged["viscosity_theoretical_cP"] = 11230 * np.exp(-0.0905 * df_merged["temperature_C"])
x = df_merged["temperature_C"]
df_merged["viscosity_theoretical_cP"] = (
    12059 + (-1283) * x + 60.3 * x**2 + (-1.51) * x**3 +
    0.0205 * x**4 + (-1.43e-4) * x**5 + 3.98e-7 * x**6)
df_merged.to_csv("theoretical_viscosity_values.csv", index=False, sep=";")

# Viscosity non-dimensionalization

# Use theoretical viscosity at lowest temperature as reference
mu0_theo = df_merged["viscosity_theoretical_cP"].iloc[0]
mu0_exp = df_merged["viscosity_cP"].iloc[0]

# Non-dimensionalize
df_merged["viscosity_exp_dimless"] = df_merged["viscosity_cP"] / mu0_exp
df_merged["viscosity_theo_dimless"] = df_merged["viscosity_theoretical_cP"] / mu0_theo

# Save CSV with non-dimensionalized viscosity
df_merged.to_csv("dimensionless_viscosity_values.csv", index=False, sep=";")

# Plot dimensionless graph
plt.figure(figsize=(10, 5))

plt.plot(df_merged["temperature_C"],
         df_merged["viscosity_exp_dimless"],
         "ro-", label="Experimental (dimless)")

plt.plot(df_merged["temperature_C"],
         df_merged["viscosity_theo_dimless"],
         "bo-", label="Theoretical (dimless)")

plt.xlabel("Temperature (°C)")
plt.ylabel("Non-dimensional viscosity (μ/μ₀)")
plt.title("Temperature vs. Non-dimensional Viscosity")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot without non-dimensionalization

plt.figure(figsize=(10, 5))

plt.scatter(df_merged["temperature_C"],  # experimental data
            df_merged["viscosity_cP"],
            c="r", linewidth=2, label="Experimental")

plt.scatter(df_merged["temperature_C"],  # theoretical curve
            df_merged["viscosity_theoretical_cP"],
            c="blue", linewidth=2, label="Theoretical")

plt.xlabel("Temperature (°C)")
plt.ylabel("Viscosity (cP)")
plt.title("Temperature vs. Viscosity")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
