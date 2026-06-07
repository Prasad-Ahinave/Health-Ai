from flask import Flask, render_template, request, redirect, send_file, session
import os
import pytesseract
from PIL import Image
import re
import json
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from flask import redirect, session
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from pdf2image import convert_from_path
# ------------------ TESSERACT ------------------
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import shutil

if shutil.which("tesseract"):
    pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")

# ------------------ APP ------------------
app = Flask(__name__)
app.secret_key = "healthai-secret"

# ------------------ LOAD SKIN CNN MODEL (ONLY ONCE) ------------------
# SKIN_MODEL = load_model("skin_model.h5")
SKIN_MODEL = load_model("skin_model.keras")

SKIN_CLASSES = [
    "Acne",
    "Eczema",
    "Psoriasis",
    "Fungal Infection",
    "Normal Skin"
]


UPLOAD_FOLDER = "uploads"
HISTORY_FILE = "history.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

# ------------------ NORMAL RANGES ------------------
NORMAL_RANGES = {
    #CBC
    "Hemoglobin": (12.0, 16.0),
    "WBC": (4000, 11000),
    "Platelets": (150000, 450000),
    "RBC": (3.8, 4.8),
    "Hematocrit": (36, 48),
    "MCV": (83, 101),
    "MCH": (26, 32),
    "MCHC": (31.5, 35.9),
    "RDW": (11.5,14.5),           #---3/13/2026 added
    "PCV": (38, 45),

    # BMP
    "Glucose": (70,100),
    "Calcium": (8.5,10.5),
    "Sodium": (135,145),
    "Potassium": (3.5,5.0),
    # "Chloride": (96,106),
    "Bicarbonate": (22,28),
    "Urea": (7,20),
    "Uric Acid": (3.5, 7.2),
    "Creatinine": (0.6,1.3),
    "BUN": (5, 21),
    "Chloride": (96, 106),
    "eGFR": (90, 200),
     
     # CMP/liver
    "Albumin": (3.5,5.0),
    "Total Protein": (6.0,8.3),
    "Bilirubin": (0.1,1.2),
    "ALT": (7,56),
    "AST": (10,40),
    # "ALP": (44,147),
    "Globulin": (2.3, 3.5),
    "A/G Ratio": (1.0, 2.3),
    "ALP": (44, 306),
    "Total Protein": (6.6, 8.3),

    #lipid
    "Total Cholesterol": (0, 200),
    "LDL": (0, 130),
    "HDL": (40, 60),
    "Triglycerides": (0, 150),
    "VLDL": (0, 40),


    # Thyroid
    "TSH": (0.4,4.0),
    "T3": (80,200),
    "T4": (5,12),

    # Iron
    "Serum Iron": (60,170),
    "Ferritin": (20,300),
    "TIBC": (240,450),

   # Vitamins
   "Vitamin B12": (200,900),
   "Vitamin D": (20,50),

   # Minerals
   "Magnesium": (1.7,2.2),
   "Zinc": (70,120),

   # Cancer
   "PSA": (0,4),
   "CA125": (0,35),
   "CA19-9": (0,37),
   "AFP": (0,10),
   "CEA": (0,5),

   # Autoimmune
   "CRP": (0,3),
   "ESR": (0,20),
   "RF": (0,14),

   # Diabetes
   "HbA1c": (4.0, 5.7)
}

# ------------------ HELPERS ------------------
# def extract_value(text, pattern):
#     match = re.search(pattern, text, re.IGNORECASE)
#     if match:
#         try:
#             return float(match.group(1))            # return float(match.group(match.lastindex))  # return float(match.group(1))
#         except:
#             return None
#     return None

def extract_value(text, pattern):
    matches = re.findall(pattern, text, re.IGNORECASE)

    if matches:
        value = matches[-1]   # take last match

        if isinstance(value, tuple):
            value = value[-1]

        return float(value)

    return None
   
   
   
   
   
    # match = re.search(pattern, text, re.IGNORECASE)
    # if match:
    #     return float(match.group(match.lastindex))
    # return None



# def analyze(test_name, value):
#     if value is None:
#       return {
#         "test": test_name,
#         "value": value,
#         "status": "Detected",
#         "suggestion": "This parameter is not included in the uploaded report."
#     }

def analyze(test_name, value):

    # If the test has no defined normal range
    if test_name not in NORMAL_RANGES:
        return {
            "test": test_name,
            "value": value,
            "status": "Detected",
            "suggestion": "No medical reference range available for this parameter."
        }

    low, high = NORMAL_RANGES[test_name]

    if value < low:
        status = "Low"
    elif value > high:
        status = "High"
    else:
        status = "Normal"
    suggestions = {
        "Hemoglobin": {
            "Low": "Low hemoglobin indicates possible anemia. Increase iron-rich foods and consult a doctor.",
            "Normal": "Hemoglobin level is normal.",
            "High": "High hemoglobin may occur due to dehydration."
        },
        "WBC": {
            "Low": "Low WBC count indicates weak immunity.",
            "Normal": "WBC count is normal.",
            "High": "High WBC may indicate infection."
        },
        "Platelets": {
            "Low": "Low platelet count increases bleeding risk.",
            "Normal": "Platelet count is normal.",
            "High": "High platelet count may increase clotting risk."
        },
        "RBC": {
            "Low": "Low RBC count may indicate anemia.",
            "Normal": "RBC count is normal.",
            "High": "High RBC count may indicate dehydration."
        },
        "Hematocrit": {
            "Low": "Low hematocrit suggests anemia.",
            "Normal": "Hematocrit is normal.",
            "High": "High hematocrit may indicate dehydration."
        },
        "MCV": {
            "Low": "Low MCV indicates microcytic anemia.",
            "Normal": "MCV is normal.",
            "High": "High MCV indicates macrocytic anemia."
        },
        "MCH": {
            "Low": "Low MCH suggests hypochromic anemia.",
            "Normal": "MCH is normal.",
            "High": "High MCH may indicate macrocytic anemia."
        },
        "MCHC": {
            "Low": "Low MCHC suggests hypochromic anemia.",
            "Normal": "MCHC is normal.",
            "High": "High MCHC may indicate blood disorder."
        },
        "Fasting Glucose": {
            "Low": "Low fasting glucose may cause hypoglycemia.",
            "Normal": "Fasting glucose is normal.",
            "High": "High fasting glucose may indicate diabetes."
        },
        "Random Glucose": {
            "Low": "Low random glucose may cause weakness.",
            "Normal": "Random glucose is normal.",
            "High": "High random glucose may indicate diabetes."
        },
        "Total Cholesterol": {
            "Low": "Low cholesterol may indicate malnutrition.",
            "Normal": "Total cholesterol is normal.",
            "High": "High cholesterol increases heart risk."
        },
        "LDL": {
            "Low": "LDL is low and safe.",
            "Normal": "LDL is normal.",
            "High": "High LDL increases heart risk."
        },
        "HDL": {
            "Low": "Low HDL increases heart risk.",
            "Normal": "HDL is healthy.",
            "High": "High HDL is beneficial."
        },
        "Triglycerides": {
            "Low": "Triglycerides are low.",
            "Normal": "Triglycerides are normal.",
            "High": "High triglycerides increase heart risk."
        },
        "TSH": {
            "Low": "Low TSH may indicate hyperthyroidism.",
            "Normal": "TSH level is normal.",
            "High": "High TSH may indicate hypothyroidism."
        },
        
        "Creatinine": {
             "Low": "Low creatinine may indicate reduced muscle mass.",
             "Normal": "Creatinine level is normal indicating proper kidney filtration.",
             "High": "High creatinine may indicate kidney dysfunction or dehydration."
          },

        "Urea": {
           "Low": "Low urea may occur due to liver disease or malnutrition.",
           "Normal": "Urea level is normal.",
           "High": "High urea may indicate kidney dysfunction or high protein breakdown."
        },

         "Sodium": {
           "Low": "Low sodium may cause fatigue, confusion or muscle weakness.",
           "Normal": "Sodium level is normal and helps maintain fluid balance.",
           "High": "High sodium may indicate dehydration or kidney issues."
        },

        "Potassium": {
          "Low": "Low potassium may cause muscle weakness or heart rhythm problems.",
          "Normal": "Potassium level is normal and supports nerve and muscle function.",
          "High": "High potassium may affect heart rhythm and requires medical attention."
        },

        "Uric Acid": {
          "Low": "Low uric acid usually does not cause serious problems.",
          "Normal": "Uric acid level is normal.",
          "High": "High uric acid may lead to gout or kidney stones."
        },

        "Albumin": {
          "Low": "Low albumin may indicate liver disease or malnutrition.",
          "Normal": "Albumin level is normal.",
          "High": "High albumin may indicate dehydration."
}   
    }
    if test_name in suggestions:
       suggestion_text = suggestions[test_name][status]

    else:
       if  status == "Normal":
           suggestion_text = "This parameter is within the healthy reference range."
       elif status == "Low":
            suggestion_text = "This parameter is lower than the normal range. Medical consultation may be recommended."
       elif status == "High":
            suggestion_text = "This parameter is higher than the normal range. Please consult a healthcare professional."

    return {
        "test": test_name,
        "value": value,
        "status": status,
        "suggestion": suggestion_text
    }

    # return {
    #     "test": test_name,
    #     "value": value,
    #     "status": status,
    #     "suggestion": suggestions[test_name][status]
    # }


# ------------------ ROUTES ------------------
@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/disclaimer")
def disclaimer():
    return render_template("disclaimer.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["blood_report"]
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        
        # text = pytesseract.image_to_string(Image.open(path)) adhi fakt hech line hoti khalychya 2 nhvtya.
        # img = Image.open(path).convert("L")
        # text = pytesseract.image_to_string(img)
        # text = ""

        # if file.filename.lower().endswith(".pdf"):

        #   pages = convert_from_path(path)

        #   for page in pages:
        #      img = page.convert("L")
        #      text += pytesseract.image_to_string(img)

        #   else:
        #      img = Image.open(path).convert("L")
        #      text = pytesseract.image_to_string(img)
          
        #   text = text.replace("\n", " ")
        #   print("OCR TEXT:\n", text) #added
        
        text = ""

        # -------- PDF REPORT --------
        if file.filename.lower().endswith(".pdf"):

        #    pages = convert_from_path(
        #    path,
        #    poppler_path=r"C:\Users\ahina\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"
        #   )
           pages = convert_from_path(path)
           for page in pages:
            img = page.convert("L")
            text += pytesseract.image_to_string(img)

        # -------- IMAGE REPORT --------
        else:
          img = Image.open(path).convert("L")
          text = pytesseract.image_to_string(img)

        # Clean OCR text
        text = text.replace("\n", " ")
        text = text.lower()
        print("OCR TEXT:\n", text)

        # tests = {
        #     "Hemoglobin": extract_value(text, r"Hemoglobin\s*L?\s*(\d+\.?\d*)"),
        #     "WBC": extract_value(text, r"WBC\s*Count\s*(\d+)"),
        #     "Platelets": extract_value(text, r"Platelet\s*Count\s*(\d+)"),
        #     "RBC": extract_value(text, r"RBC\s*Count\s*(\d+\.?\d*)"),
        #     "MCHC": extract_value(text, r"MCHC\s*(\d+\.?\d*)"),
        #     "Fasting Glucose": extract_value(text, r"(FBS|Fasting\s*Glucose)\s*(\d+\.?\d*)"),
        #     "Random Glucose": extract_value(text, r"(RBS|Random\s*Glucose)\s*(\d+\.?\d*)"),
        #     "Total Cholesterol": extract_value(text, r"Total\s*Cholesterol\s*(\d+\.?\d*)"),
        #     "LDL": extract_value(text, r"\bLDL\b\s*(\d+\.?\d*)"),
        #     "HDL": extract_value(text, r"\bHDL\b\s*(\d+\.?\d*)"),
        #     "Triglycerides": extract_value(text, r"(Triglycerides|TG)\s*(\d+\.?\d*)"),
        #     "TSH": extract_value(text, r"\bTSH\b\s*(\d+\.?\d*)")
        # }
        tests = {
             "Hemoglobin": extract_value(text, r"haemoglobin\s*\(hb\)%?\s*(\d+\.?\d*)"),  #  "Hemoglobin": extract_value(text, r"Hemoglobin\s*L?\s*(\d+\.?\d*)"),
             "WBC": extract_value(text, r"WBC\s*Count\s*L?\s*(\d+)"),
             "Platelets": extract_value(text, r"Platelet\s*Count\s*L?\s*(\d+)"),
             "RBC": extract_value(text, r"RBC\s*Count\s*(\d+\.?\d*)"),

             "Hematocrit": extract_value(text, r"Hematocrit\s*L?\s*(\d+\.?\d*)"),
             "MCV": extract_value(text, r"MCV\s*L?\s*(\d+\.?\d*)"),
             "MCH": extract_value(text, r"MCH\s*(\d+\.?\d*)"),
             "MCHC": extract_value(text, r"MCHC\s*(\d+\.?\d*)"),

            # ADDED at 3/13/2026
             "Glucose": extract_value(text, r"plasma\s*glucose\(f\)\s*(\d+\.?\d*)"), #  "Glucose": extract_value(text,r"Glucose\s*(\d+\.?\d*)"),
             "Calcium": extract_value(text,r"Calcium\s*(\d+\.?\d*)"),
             "Sodium": extract_value(text,r"Sodium\s*(\d+\.?\d*)"),
             "Potassium": extract_value(text,r"Potassium\s*(\d+\.?\d*)"),

             "Creatinine": extract_value(text,r"Creatinine\s*(\d+\.?\d*)"),
             "Urea": extract_value(text,r"Urea\s*(\d+\.?\d*)"),
             "Uric Acid": extract_value(text,r"Uric\s*Acid\s*(\d+\.?\d*)"),

             "ALT": extract_value(text, r"serum\s*sgpt\s*(\d+\.?\d*)"),#  "ALT": extract_value(text,r"(ALT|SGPT)[^\d]*(\d+\.?\d*)"),   #"ALT": extract_value(text,r"(ALT|SGPT)\s*(\d+\.?\d*)"), 
             "AST": extract_value(text, r"serum\s*sgot\s*(\d+\.?\d*)"), #  "AST": extract_value(text,r"(AST|SGOT)[^\d]*(\d+\.?\d*)"),   #"AST": extract_value(text,r"(AST|SGOT)\s*(\d+\.?\d*)"),

             "Bilirubin": extract_value(text, r"serum\s*bilirubin\s*\(\s*total\s*\)\s*(\d+\.?\d*)"),#  "Bilirubin": extract_value(text,r"Bilirubin\s*(Total)?[^\d]*(\d+\.?\d*)"),  #"Bilirubin": extract_value(text,r"Bilirubin\s*(\d+\.?\d*)"),
             "Albumin": extract_value(text,r"Albumin\s*(\d+\.?\d*)"), 

             "Vitamin B12": extract_value(text,r"Vitamin\s*B12\s*(\d+)"),
             "Vitamin D": extract_value(text,r"Vitamin\s*D\s*(\d+)"),

             "CRP": extract_value(text,r"CRP\s*(\d+\.?\d*)"),
             "ESR": extract_value(text,r"ESR\s*(\d+\.?\d*)"),

             "Glucose": extract_value(text, r"plasma\s*glucose\(f\)\s*(\d+\.?\d*)"), #  "Glucose": extract_value(text, r"(Glucose|FBS|Plasma\s*Glucose)[^\d]*(\d+\.?\d*)"),
             "HbA1c": extract_value(text, r"whole\s*blood\s*hba1c\s*(\d+\.?\d*)"),#  "HbA1c": extract_value(text, r"HbA1c[^\d]*(\d+\.?\d*)"),

             "Total Cholesterol": extract_value(text, r"serum\s*cholesterol\s*(\d+\.?\d*)"),#  "Total Cholesterol": extract_value(text, r"Cholesterol[^\d]*(\d+\.?\d*)"),
             "Triglycerides": extract_value(text, r"Triglyceride[^\d]*(\d+\.?\d*)"),
             "HDL": extract_value(text, r"serum\s*hdl-chol\s*(\d+\.?\d*)"),#  "HDL": extract_value(text, r"HDL[^\d]*(\d+\.?\d*)"),
             "LDL": extract_value(text, r"serum\s*ldl-chol\s*(\d+\.?\d*)"), #  "LDL": extract_value(text, r"LDL[^\d]*(\d+\.?\d*)"),
             "VLDL": extract_value(text, r"VLDL[^\d]*(\d+\.?\d*)"),

             "ALP": extract_value(text, r"Alkaline\s*phosphatase[^\d]*(\d+\.?\d*)"),
             "Total Protein": extract_value(text, r"Total\s*Protein[^\d]*(\d+\.?\d*)"),
             "Globulin": extract_value(text, r"Globulin[^\d]*(\d+\.?\d*)"),
             "A/G Ratio": extract_value(text, r"A/G\s*Ratio[^\d]*(\d+\.?\d*)"),

             "BUN": extract_value(text, r"BUN[^\d]*(\d+\.?\d*)"),
             "Chloride": extract_value(text, r"Chloride[^\d]*(\d+\.?\d*)"),
             "eGFR": extract_value(text, r"eGFR[^\d]*(\d+\.?\d*)"),

             "Hemoglobin": extract_value(text, r"Hb[^\d]*(\d+\.?\d*)"),
             "PCV": extract_value(text, r"pcv\s*(\d+\.?\d*)"),#  "PCV": extract_value(text, r"PCV[^\d]*(\d+\.?\d*)"),
             "RDW": extract_value(text, r"RDW[^\d]*(\d+\.?\d*)"),
             "WBC": extract_value(text, r"(TLC|Leucocyte\s*Count)[^\d]*(\d+)"),
             "Platelets": extract_value(text, r"Platelets[^\d]*(\d+)"),

    # Optional tests (may not exist in report)
    # "Fasting Glucose": extract_value(text, r"(Fasting\s*Glucose|FBS)\s*(\d+\.?\d*)"),
    # "Random Glucose": extract_value(text, r"(Random\s*Glucose|RBS)\s*(\d+\.?\d*)"),
    # "Total Cholesterol": extract_value(text, r"Total\s*Cholesterol\s*(\d+\.?\d*)"),
    # "LDL": extract_value(text, r"\bLDL\b\s*(\d+\.?\d*)"),
    # "HDL": extract_value(text, r"\bHDL\b\s*(\d+\.?\d*)"),
    # "Triglycerides": extract_value(text, r"(Triglycerides|TG)\s*(\d+\.?\d*)"),
    # "TSH": extract_value(text, r"\bTSH\b\s*(\d+\.?\d*)")
}


        # results = [analyze(k, v) for k, v in tests.items()]
        results = []

        for k, v in tests.items():
           if v is not None:
              results.append(analyze(k, v))


        with open(HISTORY_FILE, "r+") as f:
            data = json.load(f)
            data.append({"time": str(datetime.now()), "results": results})
            f.seek(0)
            json.dump(data, f, indent=2)

        session["latest"] = results
        return render_template("result.html", results=results)

    return render_template("index.html")

@app.route("/final-disclaimer")
def final_disclaimer():
    return render_template("final_disclaimer.html")

@app.route("/history")
def history():
    with open(HISTORY_FILE) as f:
        data = json.load(f)
    return render_template("history.html", history=data)

@app.route("/download-pdf")
def download_pdf():
    pdf_path = "report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=A4)
    y = 800
    c.drawString(50, y, "Blood Report Analysis")
    y -= 40

    for item in session.get("latest", []):
        c.drawString(50, y, f"{item['test']} : {item['value']} ({item['status']})")
        y -= 20

    c.save()
    return send_file(pdf_path, as_attachment=True)


#-------------Skin Route---------------
@app.route("/skin")
def skin_upload():
    return render_template("skin_upload.html")

@app.route("/skin-analyze", methods=["POST"])
def skin_analyze():
    file = request.files["skin_image"]

    if file.filename == "":
        return "No image selected"

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    # Load & preprocess image
    img = image.load_img(path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # CNN prediction
    predictions = SKIN_MODEL.predict(img_array)
    confidence = float(np.max(predictions)) * 100
    label_index = int(np.argmax(predictions))
    label = SKIN_CLASSES[label_index]

    skin_info = {
        "Acne": (
            "Acne is a common skin condition caused by clogged pores.",
            "Keep skin clean and consult a dermatologist if severe."
        ),
        "Eczema": (
            "Eczema causes dry, itchy, and inflamed skin.",
            "Moisturize regularly and avoid allergens."
        ),
        "Psoriasis": (
            "Psoriasis is a chronic autoimmune skin condition.",
            "Avoid triggers and consult a dermatologist."
        ),
        "Fungal Infection": (
            "Fungal infections are caused by fungal overgrowth.",
            "Keep the area dry and use antifungal treatment."
        ),
        "Normal Skin": (
            "No major skin disease detected.",
            "Maintain proper skin hygiene."
        )
    }

    explanation, suggestion = skin_info[label]

    result = {
        "label": label,
        "confidence": round(confidence, 2),
        "explanation": explanation,
        "suggestion": suggestion
    }

    return render_template("skin_result.html", result=result)


# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(debug=True)
