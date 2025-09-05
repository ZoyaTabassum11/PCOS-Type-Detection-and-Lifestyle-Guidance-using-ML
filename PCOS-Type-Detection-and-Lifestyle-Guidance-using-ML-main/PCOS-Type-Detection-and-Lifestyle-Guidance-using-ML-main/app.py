import os
import re
import io
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
import numpy as np # Import numpy for np.nan
from sklearn.metrics import accuracy_score, classification_report
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure Flask session secret key (IMPORTANT! Change this for production)
app.config['SECRET_KEY'] = 'your_super_secret_and_long_key_here_please_change_me' # <<<<<<< IMPORTANT: Change this!

# Define allowed file extensions for uploads (only images now)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'} # Removed 'pdf'

# Make sure this path points to the tesseract.exe executable, not the installer.
# It's typically within the Tesseract-OCR folder after installation, e.g., 'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Lenovo\AppData\Local\Programs\Tesseract-OCR\tesseract.exe' 

# Load the dataset
# Ensure 'final_pcos_dataset.csv' is in the same directory as app.py
try:
    file_path = 'final_pcos_dataset.csv'
    pcos_data = pd.read_csv(file_path)

    # Preprocess the data
    # Ensure these columns exist in your CSV
    # Make a copy to avoid SettingWithCopyWarning
    pcos_data_clean = pcos_data.drop(columns=["Sl. No", "Patient File No.", "PCOS (Y/N)","Pulse rate(bpm)","RR(breaths/min)","Waist:Hip Ratio","No. of abortions","BP_Systolic(mmHg)","BP_Diastolic(mmHg)"]).copy()
    pcos_data_clean = pcos_data_clean.apply(pd.to_numeric, errors='coerce')
    pcos_data_clean['PCOS_Type'] = pcos_data['PCOS_Type'] # Assigning original PCOS_Type column
    
    label_encoder = LabelEncoder()
    # Ensure 'PCOS_Type' column exists and is handled. If it's categorical string, encode it.
    # Assuming 'PCOS_Type' is what you want to predict and it contains string labels.
    pcos_data_clean['PCOS_Type'] = label_encoder.fit_transform(pcos_data_clean['PCOS_Type'])
    
    imputer = SimpleImputer(strategy='median')
    # Apply imputer only to numeric columns before dropping 'PCOS_Type' for X
    numeric_cols = pcos_data_clean.select_dtypes(include=['number']).columns.drop('PCOS_Type', errors='ignore')
    pcos_data_clean[numeric_cols] = imputer.fit_transform(pcos_data_clean[numeric_cols])

    # Define features (X) and target (y)
    X = pcos_data_clean.drop(columns=['PCOS_Type'])
    y = pcos_data_clean['PCOS_Type']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    model_loaded = True
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # print(f"\n✅ Model trained successfully!")
    # print(f"📊 Accuracy on test data: {acc:.2f}")
    # print("📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
except Exception as e:
    print(f"Error loading or training model: {e}")
    model_loaded = False
# print(model_loaded)

categorical_mappings = {
    "Blood Group": {1: "A+", 2: "A-", 3: "B+", 4: "B-", 5: "O+", 6: "O-", 7: "AB+", 8: "AB-"},
    "Cycle(R/I)": {1: "Regular", 0: "Irregular"},
    "Marraige Status(Yrs)": {0: "Not Married", 1: "Married"},
    "Pregnant(Y/N)": {0: "No", 1: "Yes"},
    "Weight gain(Y/N)": {0: "No", 1: "Yes"},
    "hair growth(Y/N)": {0: "No", 1: "Yes"},
    "Skin darkening(Y/N)": {0: "No", 1: "Yes"},
    "Hair loss(Y/N)": {0: "No", 1: "Yes"},
    "Pimples(Y/N)": {0: "No", 1: "Yes"},
    "Fast food(Y/N)": {0: "No", 1: "Yes"},
    "Reg.Exercise(Y/N)": {0: "No", 1: "Yes"}
}

# Define valid ranges for numeric features (from your provided app.py)
feature_ranges = {
    "Age(yrs)": (10, 50),
    "Weight(Kg)": (30, 150),
    "Height(Cm)": (120, 210),
    "BMI": (10, 60),
    "Blood Group": (1, 8),
    # "Pulse rate(bpm)": (15, 120),
    # "RR(breaths/min)": (10, 30),
    "Hb(g/dl)": (7, 18),
    "Cycle(R/I)": (0, 1),
    "Cycle length(days)": (20, 45),
    "Marraige Status(Yrs)": (0, 1),
    "Pregnant(Y/N)": (0, 1),
    # "No. of abortions": (0, 5),
    "I beta-HCG(mIU/mL)": (0, 50000),
    "II beta-HCG(mIU/mL)": (0, 50000),
    "FSH(mIU/mL)": (0, 40),
    "LH(mIU/mL)": (0, 20),
    "FSH/LH": (0, 2),
    "Hip(inch)": (20, 60),
    "Waist(inch)": (20, 60),
    # "Waist:Hip Ratio": (0.5, 1.0),
    "TSH(mIU/L)": (0.3, 5.5),
    "AMH(ng/mL)": (0.1, 20),
    "PRL(ng/mL)": (1, 25),
    "Vit D3(ng/mL)": (10, 50),
    "PRG(ng/mL)": (1, 10),
    "RBS(mg/dl)": (70, 500),
    "Weight gain(Y/N)": (0, 1),
    "hair growth(Y/N)": (0, 1),
    "Skin darkening(Y/N)": (0, 1),
    "Hair loss(Y/N)": (0, 1),
    "Pimples(Y/N)": (0, 1),
    "Fast food(Y/N)": (0, 1),
    "Reg.Exercise(Y/N)": (0, 1),
    # "BP_Systolic(mmHg)": (80, 200),
    # "BP_Diastolic(mmHg)": (40, 120),
    "Follicle No.(L)": (0, 20),
    "Follicle No.(R)": (0, 20),
    "Avg.F size(L)(mm)": (0, 30),
    "Avg.F size(R)(mm)": (0, 30),
    "Endometrium(mm)": (1, 20),
}


# After model training

# Evaluate model performance on the test set



def generate_pcos_plan(pcos_type):
    # Define lifestyle guidance for different PCOS types (from your provided app.py)
    plans = {
        "Insulin-Resistant": {
            "Diet": {
                "Details": "Focus on low-glycemic index foods, high fiber, and lean proteins. Avoid processed sugars and refined carbs.",
                "Meal Plan": {
                    "Breakfast": "Greek yogurt with berries and a sprinkle of chia seeds.",
                    "Lunch": "Grilled chicken salad with quinoa and avocado.",
                    "Dinner": "Baked salmon with roasted vegetables.",
                    "Snack": "Handful of mixed nuts or an apple with almond butter."
                }
            },
            "Exercise": {
                "Details": "Moderate-intensity cardio (e.g., brisk walking, cycling) 4-5 times a week, strength training 2-3 times a week.",
            },
            "Supplements": {
                "Details": [
                    "Inositol: Supports insulin sensitivity and hormonal balance. Recommended dosage: 2-4 grams daily.",
                    "Omega-3 fatty acids: Reduces inflammation and improves metabolic health. Recommended dosage: 1-2 grams daily.",
                    "Vitamin D: Helps regulate menstrual cycles and supports bone health. Recommended dosage: 1000-2000 IU daily."
                ]
            },
            "Stress Management": {
                "Details": "Practice mindfulness meditation for 10 minutes daily. Ensure 7-8 hours of sleep.",
            }
        },
        "Inflammatory": {
            "Diet": {
                "Details": "Follow an anti-inflammatory diet including berries, fatty fish, green tea, and turmeric. Avoid dairy and gluten if sensitive.",
                "Meal Plan": {
                    "Breakfast": "Smoothie with spinach, berries, almond milk, and a scoop of plant-based protein.",
                    "Recipe":"https://example.com/smoothie-recipe",
                    "Lunch": "Grilled chicken with roasted sweet potatoes and steamed broccoli.",
                    "Recipe":"https://example.com/chicken-recipe",
                    "Dinner": "Zucchini noodles with pesto and grilled shrimp.",
                    "Recipe":"https://example.com/zoodles-recipe",
                    "Snack": "A small handful of walnuts or a green tea latte.",
                    "Recipe":"https://example.com/snack-ideas",
                }
            },
            "Exercise": {
                "Details": "Low to moderate-intensity exercises like swimming or hiking, 4-5 times a week.",
                "Videos": [
                    "http://www.youtube.com/watch?v=1NGpG3kyPc4", # Swimming Basics for Beginners
                    "http://www.youtube.com/watch?v=Ul9ryQiK8VM", # Hiking Tips and Techniques
                ]
            },
            "Supplements": {
                "Details": [
                    "Curcumin: Reduces inflammation. Recommended dosage: 500-1000 mg daily.",
                    "Zinc: Supports immune function and reduces inflammation. Recommended dosage: 8-12 mg daily.",
                    "Probiotics: Promotes gut health. Recommended dosage: 1-10 billion CFUs daily."
                ]
            },
            "Stress Management": {
                "Details": "Journaling or art therapy to reduce stress levels. Focus on gratitude practices.",
                "Videos": [
                    "http://www.youtube.com/watch?v=MXITTbeLDfA", # How to Start Journaling
                    "http://www.youtube.com/watch?v=nA5dGCeZO5k" # Art Therapy for Stress Relief
                ]
            }
        },
        "Post-Pill": {
            "Diet": {
                "Details": "Balanced diet emphasizing whole foods and plenty of vegetables. Avoid extreme dieting or restrictive eating.",
                "Meal Plan": {
                    "Breakfast": "Scrambled eggs with spinach and whole-grain toast.",
                    "Lunch": "Quinoa salad with chickpeas, cucumber, and feta.",
                    "Dinner": "Roast chicken with steamed asparagus and wild rice.",
                    "Snack": "Hummus with carrot and celery sticks."
                }
            },
            "Exercise": {
                "Details": "Gentle to moderate exercise, such as light cardio and bodyweight strength exercises.",
                "Videos": [
                    "http://www.youtube.com/watch?v=-yMkmCGkwXo", # Light Cardio for Beginners
                    "http://www.youtube.com/watch?v=Pt5BJO9zA8s" # Bodyweight Strength Workout
                ]
            },
            "Supplements": {
                "Details": [
                    "Zinc: Supports skin health and hormonal balance. Recommended dosage: 8-12 mg daily.",
                    "Vitamin B Complex: Boosts energy and supports hormonal function. Recommended dosage: As directed on supplement.",
                    "Adaptogens: Helps the body adapt to stress. Recommended dosage: Follow product instructions."
                ]
            },
            "Stress Management": {
                "Details": "Schedule downtime and incorporate a hobby you enjoy.",
                "Videos": [
                    "http://www.youtube.com/watch?v=grfXR6FAsI8", # How to Manage Stress with Hobbies
                    "http://www.youtube.com/watch?v=DbDoBzGY3vo" # Relaxation Techniques for Beginners
                ]
            }
        },
        "Adrenal": {
            "Diet": {
                "Details": "Increase magnesium and Vitamin B-rich foods like nuts, seeds, leafy greens, and whole grains. Avoid stimulants like caffeine.",
                "Meal Plan": {
                    "Breakfast": "Oatmeal with almond milk, banana slices, and a sprinkle of flax seeds.",
                    "Lunch": "Stir-fried tofu and vegetables with brown rice.",
                    "Dinner": "Sweet potato and black bean tacos with avocado.",
                    "Snack": "A small handful of pumpkin seeds or dark chocolate (85% cocoa)."
                }
            },
            "Exercise": {
                "Details": "Low-intensity exercises such as yoga, pilates, or walking, focusing on stress relief.",
                "Videos": [
                    "http://www.youtube.com/watch?v=EvMTrP8eRvM", # Gentle Yoga for Stress Relief
                    "http://www.youtube.com/watch?v=bO6NNfX_1ns" # Walking Workout for Beginners
                ]
            },
            "Supplements": {
                "Details": [
                    "Ashwagandha: Supports adrenal health and reduces stress. Recommended dosage: 300-600 mg daily.",
                    "Rhodiola: Enhances stress tolerance and energy levels. Recommended dosage: 200-400 mg daily.",
                    "Magnesium: Helps reduce muscle tension and improve sleep. Recommended dosage: 300-400 mg daily."
                ]
            },
            "Stress Management": {
                "Details": "Include deep breathing exercises, and establish a consistent bedtime routine.",
                "Videos": [
                    "http://www.youtube.com/watch?v=F28MGLlpP90", # Deep Breathing Techniques
                    "http://www.youtube.com/watch?v=Gr031jDm_00" # Bedtime Routine for Better Sleep
                ]
            }
        }
    }
    plan = plans.get(pcos_type, "PCOS type not recognized. Provide a valid type.")
    return plan

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_info_from_document(filepath):
    """
    Extracts text from an image and attempts to identify PCOS form fields.
    This uses rule-based extraction and will require fine-tuning for real-world reports.
    """
    extracted_data = {}
    full_text = ""
    file_extension = filepath.rsplit('.', 1)[1].lower()

    if file_extension in ALLOWED_EXTENSIONS: # Checks against the updated ALLOWED_EXTENSIONS
        try:
            img = Image.open(filepath)
            full_text = pytesseract.image_to_string(img)
        except Exception as e:
            print(f"Error processing image {filepath} with OCR: {e}")
            extracted_data['error'] = f"Image OCR failed: {e}"
            return extracted_data
    else:
        # This case should ideally be caught by allowed_file before reaching here,
        # but as a safeguard, mark an error for disallowed file types.
        extracted_data['error'] = "Unsupported file type for OCR."
        return extracted_data


    print(f"--- Full Extracted Text ---\n{full_text}\n--- End Extracted Text ---")

    # Helper function to safely extract and convert values
    def safe_extract_and_convert(pattern, text, target_type=str, group=1):
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            value = match.group(group).strip()
            try:
                if target_type == int:
                    # Clean non-digit characters for integers
                    clean_value = re.sub(r'[^\d]', '', value)
                    return int(clean_value) if clean_value else None
                elif target_type == float:
                    # Clean non-digit and non-dot characters for floats
                    clean_value = re.sub(r'[^\d.]', '', value)
                    return float(clean_value) if clean_value else None
                return target_type(value)
            except ValueError:
                return None # Return None if conversion fails
        return None # Return None if no match

    # --- Information Extraction Logic for PCOS Form Fields ---
    # Numeric fields
    extracted_data['Age(yrs)'] = safe_extract_and_convert(r'Age\s*\(yrs\):\s*(\d+)', full_text, int)
    extracted_data['Weight(Kg)'] = safe_extract_and_convert(r'Weight\s*\(Kg\):\s*(\d+\.?\d*)', full_text, float)
    extracted_data['Height(Cm)'] = safe_extract_and_convert(r'Height\s*\(Cm\):\s*(\d+\.?\d*)', full_text, float)
    extracted_data['BMI'] = safe_extract_and_convert(r'BMI:\s*(\d+\.?\d*)', full_text, float)
    extracted_data['Hb(g/dl)'] = safe_extract_and_convert(r'Hb\s*\(g/dl\):\s*(\d+\.?\d*)', full_text, float)
    extracted_data['Cycle length(days)'] = safe_extract_and_convert(r'Cycle\s*Length\s*\(days\):\s*(\d+)', full_text, int)
    extracted_data['Marraige Status(Yrs)'] = safe_extract_and_convert(r'Marriage\s*Status\s*\(Yrs\):\s*(\d+)', full_text, int)
    extracted_data['I beta-HCG(mIU/mL)'] = safe_extract_and_convert(r'I\s*beta-HCG\s*\(mIU/mL\):\s*(\d+\.?\d*)', full_text, float)
    extracted_data['II beta-HCG(mIU/mL)'] = safe_extract_and_convert(r'II\s*beta-HCG\s*\(mIU/mL\):\s*(\d+\.?\d*)', full_text, float)
    extracted_data['FSH(mIU/mL)'] = safe_extract_and_convert(r'FSH\s*\(mIU/mL\):\s*(\d+\.?\d*)', full_text, float)
    extracted_data['LH(mIU/mL)'] = safe_extract_and_convert(r'LH\s*\(mIU/mL\):\s*(\d+\.?\d*)', full_text, float)
    extracted_data['FSH/LH'] = safe_extract_and_convert(r'FSH/LH:\s*(\d+\.?\d*)', full_text, float)
    extracted_data['Hip(inch)'] = safe_extract_and_convert(r'Hip\s*\(inches\):\s*(\d+\.?\d*)', full_text, float)
    extracted_data['Waist(inch)'] = safe_extract_and_convert(r'Waist\s*\(inches\):\s*(\d+\.?\d*)', full_text, float)
    extracted_data['TSH(mIU/L)'] = safe_extract_and_convert(r'TSH\s*\(mIU/L\):\s*(\d+\.?\d*)', full_text, float)
    extracted_data['AMH(ng/mL)'] = safe_extract_and_convert(r'AMH\s*\(ng/mL\):\s*(\d+\.?\d*)', full_text, float)
    extracted_data['PRL(ng/mL)'] = safe_extract_and_convert(r'PRL\s*\(ng/mL\):\s*(\d+\.?\d*)', full_text, float)
    extracted_data['Vit D3(ng/mL)'] = safe_extract_and_convert(r'Vitamin\s*D3\s*\(ng/mL\):\s*(\d+\.?\d*)', full_text, float)
    extracted_data['PRG(ng/mL)'] = safe_extract_and_convert(r'PRG\s*\(ng/mL\):\s*(\d+\.?\d*)', full_text, float)
    extracted_data['RBS(mg/dl)'] = safe_extract_and_convert(r'RBS\s*\(mg/dl\):\s*(\d+\.?\d*)', full_text, float)
    extracted_data['Follicle No.(L)'] = safe_extract_and_convert(r'Follicle\s*Number\s*\(Left\):\s*(\d+)', full_text, int)
    extracted_data['Follicle No.(R)'] = safe_extract_and_convert(r'Follicle\s*Number\s*\(Right\):\s*(\d+)', full_text, int)
    extracted_data['Avg.F size(L)(mm)'] = safe_extract_and_convert(r'Average\s*Follicle\s*Size\s*\(Left\)\s*\(mm\):\s*(\d+\.?\d*)', full_text, float)
    extracted_data['Avg.F size(R)(mm)'] = safe_extract_and_convert(r'Average\s*Follicle\s*Size\s*\(Right\)\s*\(mm\):\s*(\d+\.?\d*)', full_text, float)
    extracted_data['Endometrium(mm)'] = safe_extract_and_convert(r'Endometrium\s*\(mm\):\s*(\d+\.?\d*)', full_text, float)

    # Yes/No or Regular/Irregular fields (map text to 0 or 1)
    extracted_data['Cycle(R/I)'] = "1" if re.search(r'Cycle\s*\(Regular/Irregular\):\s*Regular', full_text, re.IGNORECASE) else "0" if re.search(r'Cycle\s*\(Regular/Irregular\):\s*Irregular', full_text, re.IGNORECASE) else None
    extracted_data['Pregnant(Y/N)'] = "1" if re.search(r'Pregnant\s*\(Yes/No\):\s*Yes', full_text, re.IGNORECASE) else "0" if re.search(r'Pregnant\s*\(Yes/No\):\s*No', full_text, re.IGNORECASE) else None
    extracted_data['Weight gain(Y/N)'] = "1" if re.search(r'Weight\s*Gain\s*\(Y/N\):\s*Yes', full_text, re.IGNORECASE) else "0" if re.search(r'Weight\s*Gain\s*\(Y/N\):\s*No', full_text, re.IGNORECASE) else None
    extracted_data['hair growth(Y/N)'] = "1" if re.search(r'Hair\s*Growth\s*\(Y/N\):\s*Yes', full_text, re.IGNORECASE) else "0" if re.search(r'Hair\s*Growth\s*\(Y/N\):\s*No', full_text, re.IGNORECASE) else None
    extracted_data['Skin darkening(Y/N)'] = "1" if re.search(r'Skin\s*Darkening\s*\(Y/N\):\s*Yes', full_text, re.IGNORECASE) else "0" if re.search(r'Skin\s*Darkening\s*\(Y/N\):\s*No', full_text, re.IGNORECASE) else None
    extracted_data['Hair loss(Y/N)'] = "1" if re.search(r'Hair\s*Loss\s*\(Y/N\):\s*Yes', full_text, re.IGNORECASE) else "0" if re.search(r'Hair\s*Loss\s*\(Y/N\):\s*No', full_text, re.IGNORECASE) else None
    extracted_data['Pimples(Y/N)'] = "1" if re.search(r'Pimples\s*\(Y/N\):\s*Yes', full_text, re.IGNORECASE) else "0" if re.search(r'Pimples\s*\(Y/N\):\s*No', full_text, re.IGNORECASE) else None
    extracted_data['Fast food(Y/N)'] = "1" if re.search(r'Fast\s*Food\s*\(Y/N\):\s*Yes', full_text, re.IGNORECASE) else "0" if re.search(r'Fast\s*Food\s*\(Y/N\):\s*No', full_text, re.IGNORECASE) else None
    extracted_data['Reg.Exercise(Y/N)'] = "1" if re.search(r'Regular\s*Exercise\s*\(Y/N\):\s*Yes', full_text, re.IGNORECASE) else "0" if re.search(r'Regular\s*Exercise\s*\(Y/N\):\s*No', full_text, re.IGNORECASE) else None

    # Blood Group (map text to numerical value as per HTML options)
    blood_group_text = safe_extract_and_convert(r'Blood\s*Group:\s*(A\+|A-|B\+|B-|O\+|O-|AB\+|AB-)', full_text, str)
    blood_group_map = {
        'A+': '1', 'A-': '2', 'B+': '3', 'B-': '4',
        'O+': '5', 'O-': '6', 'AB+': '7', 'AB-': '8'
    }
    extracted_data['Blood Group'] = blood_group_map.get(blood_group_text, None)

    # Filter out None values and convert them to empty strings for form pre-filling
    for key, value in extracted_data.items():
        if value is None:
            extracted_data[key] = "" # Set to empty string for HTML input value

    return extracted_data


@app.route('/', methods=['GET'])
def index():
    # Check if there's extracted data in the session
    form_data_from_session = session.pop('extracted_form_data', {})
    errors_from_session = session.pop('upload_errors', [])

    # Initial render of the form, pre-filling with session data if available
    return render_template('index.html',
                           form_data=form_data_from_session, # Use session data to pre-fill
                           errors=errors_from_session,
                           prediction_text=None,
                           plan=None,
                           categorical_mappings=categorical_mappings)

@app.route('/upload', methods=['POST'])
def upload_report():
    if not model_loaded:
        session['upload_errors'] = ["Model could not be loaded. Please check 'final_pcos_dataset.csv'."]
        return redirect(url_for('index'))

    if 'report_file' not in request.files:
        session['upload_errors'] = ["No file part in the request."]
        return redirect(url_for('index'))

    report_file = request.files['report_file']

    if report_file.filename == '':
        session['upload_errors'] = ["No selected file."]
        return redirect(url_for('index'))

    if report_file and allowed_file(report_file.filename):
        filename = secure_filename(report_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        report_file.save(filepath)

        extracted_info = extract_info_from_document(filepath)
        if 'error' in extracted_info:
            session['upload_errors'] = [f"Error processing uploaded file: {extracted_info['error']}"]
            return redirect(url_for('index'))
        else:
            # Store extracted data in session for the main form to pick up
            session['extracted_form_data'] = extracted_info
            # No errors from upload, clear previous upload errors
            session['upload_errors'] = []
            return redirect(url_for('index'))
    else:
        session['upload_errors'] = ["Invalid file type or no file selected. Allowed types: " + ", ".join(ALLOWED_EXTENSIONS) + "."]
        return redirect(url_for('index'))


@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return render_template('index.html', errors=["Model could not be loaded. Please check 'final_pcos_dataset.csv'."],
                               form_data=request.form.to_dict(), prediction_text=None, plan=None, categorical_mappings=categorical_mappings)

    form_data_for_template = request.form.to_dict() # Get all current form data
    user_data = []
    errors = []

    for col in X.columns: # Iterate through model's expected columns
        value_from_form = form_data_for_template.get(col)

        if value_from_form is None or value_from_form == "":
            errors.append(f"Missing value for {col}.")
            user_data.append(np.nan) # Append NaN for missing numeric values to be imputed later
            continue

        if col in categorical_mappings:
            try:
                int_value = int(value_from_form)
                # Check if the integer value maps to a valid category
                if int_value in categorical_mappings[col]:
                    user_data.append(int_value)
                else:
                    errors.append(f"Invalid value for {col}. Please choose a valid option (e.g., for Blood Group: 1-8).")
                    user_data.append(np.nan) # Treat invalid categorical as NaN for potential imputation if desired
            except ValueError:
                errors.append(f"Invalid input for {col}. Please enter a valid integer.")
                user_data.append(np.nan) # Treat non-integer categorical as NaN
        else: # Numeric features
            try:
                float_value = float(value_from_form)
                min_val, max_val = feature_ranges[col]
                if min_val <= float_value <= max_val:
                    user_data.append(float_value)
                else:
                    errors.append(f"Invalid value for {col}. Please enter a value between {min_val} and {max_val}.")
                    user_data.append(np.nan) # Append NaN for out-of-range numeric values
            except ValueError:
                errors.append(f"Invalid input for {col}. Please enter a valid number.")
                user_data.append(np.nan) # Treat non-numeric as NaN

    # Before creating DataFrame for prediction, ensure all critical features have valid (non-None/NaN) data
    # Now that we're using np.nan, check for np.nan in user_data instead of None
    if np.nan in user_data or errors:
        # If any required field is NaN after processing, add a generic error
        if np.nan in user_data and "One or more required fields have invalid or missing data for prediction." not in errors:
            errors.append("One or more required fields have invalid or missing data for prediction.")
        return render_template('index.html', errors=errors, form_data=form_data_for_template,
                               prediction_text=None, plan=None, categorical_mappings=categorical_mappings)

    try:
        # Create DataFrame from user_data ensuring column order matches X.columns
        user_data_df = pd.DataFrame([user_data], columns=X.columns)
        
        # Apply the same imputer used during training to handle any potential NaNs in user_data_df
        # If any `np.nan` values are present, they will be imputed.
        numeric_cols_for_imputation = user_data_df.select_dtypes(include=['number']).columns
        if not numeric_cols_for_imputation.empty:
            user_data_df[numeric_cols_for_imputation] = imputer.transform(user_data_df[numeric_cols_for_imputation])


        prediction = rf_model.predict(user_data_df)
        predicted_type = label_encoder.inverse_transform(prediction)[0]
        plan = generate_pcos_plan(predicted_type)

        return render_template('index.html',
                               prediction_text=f'Predicted PCOS Type: {predicted_type}',
                               form_data=form_data_for_template, # Pass the filled form data back
                               plan=plan,
                               categorical_mappings=categorical_mappings,
                               errors=[]) # Clear errors on successful prediction
    except Exception as e:
        errors.append(f"An error occurred during prediction: {e}")
        return render_template('index.html', errors=errors, form_data=form_data_for_template,
                               prediction_text=None, plan=None, categorical_mappings=categorical_mappings)
if __name__ == "__main__":
    app.run(debug=True)