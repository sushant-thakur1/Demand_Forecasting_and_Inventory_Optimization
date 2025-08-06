from flask import Flask, render_template, request
import numpy as np
import pickle
import traceback

app = Flask(__name__)

# Load the model once when the application starts
try:
    with open('lightgbm.pkl', 'rb') as file:
        regressor = pickle.load(file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()
    regressor = None

# A mapping for one-hot encoded and ordinal features
FEATURE_MAPPING = {
    'Manufacturer': ['Private Label', 'General MI', 'Frito Lay', 'P&G', 'Kellogg', 'Tombstone', 'Tonys', 'Snyders', 'Warner'],
    'Category': ['Bag Snacks', 'Oral Hygiene', 'Cold Cereal', 'Frozen Pizza'],
    'Sub_Category': ['Pretzels', 'Mouthwashes(Antiseptic)', 'All Family Cereal', 'Adult Cereal', 'Pizza/Premium', 'Mouthwash/Rinses And Sprays', 'Kids Cereal'],
    'Province_Code': ['KY(Kentucky)', 'TX(Texas)', 'OH(Ohio)', 'IN(India)'],
    'MSA_CODE': ['17140', '19100', '26420', '17780', '47540', '43300', '13140', '19380', '44220'],
    'SEG_VALUE_NAME': {'Value': 1, 'Mainstream': 2, 'Upscale': 3},
}

def get_encoded_features(form_data):
    """
    Processes form data and returns a list of features in the correct order.
    """
    temp_array = []
    
    # Numerical Features
    numerical_features = ['BASE_PRICE', 'PRODUCT_SIZE', 'SALES_AREA_SIZE_NUM', 'AVG_WEEKLY_BASKETS']
    for feature in numerical_features:
        value = form_data.get(feature)
        # Using a default of 0 and casting to the correct type
        temp_array.append(float(value) if feature in ['BASE_PRICE', 'PRODUCT_SIZE'] else int(value))

    # Binary Features
    temp_array.append(1 if form_data.get('FEATURE') == 'ON_Feature' else 0)
    temp_array.append(1 if form_data.get('DISPLAY') == 'ON_Display' else 0)
    
    # One-Hot Encoded Features
    for feature_name, categories in FEATURE_MAPPING.items():
        if feature_name in ['SEG_VALUE_NAME']:
            continue # Skip ordinal features
        
        selected_category = form_data.get(feature_name)
        ohe_vector = [1 if cat == selected_category else 0 for cat in categories]
        temp_array.extend(ohe_vector)

    # Ordinal Feature (must be last based on your original code's order)
    seg_value = form_data.get('SEG_VALUE_NAME')
    temp_array.append(FEATURE_MAPPING['SEG_VALUE_NAME'].get(seg_value, 0))

    return temp_array

@app.route("/", methods=["GET"])
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    if regressor is None:
        return render_template("result.html", my_prediction="Error: Model could not be loaded.")
    
    try:
        temp_array = get_encoded_features(request.form)
        
        # Verify the feature count
        expected_features = 40
        if len(temp_array) != expected_features:
            error_msg = f"Input feature count mismatch: Expected {expected_features}, got {len(temp_array)}."
            return render_template("result.html", my_prediction=error_msg)

        data = np.array([temp_array])
        my_prediction = int(regressor.predict(data)[0])
        
        return render_template("result.html", my_prediction=my_prediction)
        
    except (ValueError, TypeError) as e:
        return render_template("result.html", my_prediction=f"Input Error: All fields must be filled out with valid data. {str(e)}")
    except Exception as e:
        return render_template("result.html", my_prediction=f"An unexpected error occurred: {str(e)}")

if __name__ == '__main__':
    # app.run(debug=True, port=4500)
    app.run(host='0.0.0.0',port=80)