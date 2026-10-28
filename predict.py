"""
Inference Script for Rice Leaf Disease Detection.
Takes an input image, applies the research-based preprocessing and feature extraction pipeline,
and predicts the disease class with confidence score and agronomic recommendations.

Usage:
    python predict.py --image "Rice Leaf Disease Images/Blast/BLAST4_114.jpg"
    python predict.py --image "Dataset/Test_Data_SVM/tungro/tungro__0_9432.jpg" --model saved_models/best_model_ds2.joblib
"""

import os
import sys
import argparse
import joblib
import numpy as np

# Ensure local modules are accessible
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import SAVED_MODELS_DIR, ML_LEVELS
from src.preprocessing import preprocess_image
from src.feature_extraction import extract_fused_features

# Agronomic Knowledge Base for Rice Crop Diseases
DISEASE_ADVISORY = {
    'bacterialblight': {
        'display_name': 'Bacterial Leaf Blight (Xanthomonas oryzae)',
        'symptoms': 'Water-soaked to yellowish-white stripes along leaf margins, progressing to wavy lesions and wilting ("kresek").',
        'causes': 'Bacterial infection favored by high humidity (70-90%), warm temperatures (25-34°C), and excess nitrogen fertilizer.',
        'treatment': '1. Apply copper-based bactericides (e.g., Copper Oxychloride) or Streptomycin sulfate.\n2. Ensure balanced potassium and avoid excessive nitrogen fertilizer.\n3. Drain floodwaters temporarily to lower humidity in the field canopy.'
    },
    'blight': {
        'display_name': 'Bacterial Leaf Blight (Xanthomonas oryzae)',
        'symptoms': 'Water-soaked lesions on leaf margins turning grayish-white with yellow borders.',
        'causes': 'Systemic bacterial transmission via rain splash, irrigation canals, and strong winds.',
        'treatment': '1. Spray bactericide copper hydroxide or validamycin.\n2. Practice crop rotation and use certified disease-free seeds.'
    },
    'blast': {
        'display_name': 'Rice Blast (Magnaporthe oryzae / Pyricularia oryzae)',
        'symptoms': 'Spindle-shaped or diamond-shaped lesions with gray/white centers and reddish-brown borders on leaves and neck.',
        'causes': 'Fungal pathogen spreading rapidly during prolonged leaf wetness, cool nights, and high nitrogen application.',
        'treatment': '1. Spray systemic fungicides: Tricyclazole 75 WP (0.6 g/L) or Isoprothiolane 40 EC (1.5 mL/L).\n2. Avoid dense planting to improve air circulation.\n3. Apply balanced NPK fertilizers (do not over-apply nitrogen).'
    },
    'brownspot': {
        'display_name': 'Brown Spot (Bipolaris oryzae / Cochliobolus miyabeanus)',
        'symptoms': 'Small, circular to oval dark brown spots across leaves, often with a yellow halo.',
        'causes': 'Fungal infection associated with nutrient-deficient, silica-poor, or water-stressed soils ("poor man\'s disease").',
        'treatment': '1. Spray Mancozeb (2 g/L) or Carbendazim (1 g/L) or Propiconazole.\n2. Improve soil fertility by adding compost, silicon, and balanced potash (potassium).\n3. Treat seeds with Carbendazim before sowing.'
    },
    'tungro': {
        'display_name': 'Rice Tungro Disease (RTBV & RTSV Viral Complex)',
        'symptoms': 'Distinct yellowing or orange-yellow discoloration of leaves starting from tips, accompanied by severe plant stunting.',
        'causes': 'Viral transmission by Green Leafhopper (Nephotettix virescens) vectors.',
        'treatment': '1. Control green leafhopper vectors by applying Imidacloprid (0.3 mL/L) or Thiamethoxam.\n2. Rogue (uproot and destroy) infected plants immediately.\n3. Plant synchronized crops and resistant rice varieties (e.g., IR36, IR64).'
    },
    'sheath_blight': {
        'display_name': 'Sheath Blight (Rhizoctonia solani)',
        'symptoms': 'Oval or irregular greenish-gray water-soaked spots on leaf sheaths near water line, progressing to blotches with dark brown margins.',
        'causes': 'Soil-borne fungal sclerotia floating on floodwater during warm, humid conditions.',
        'treatment': '1. Apply Hexaconazole 5% SC (2 mL/L) or Azoxystrobin + Difenoconazole.\n2. Keep water levels low during tillering stage.\n3. Remove weed hosts on bunds.'
    },
    'leafsmut': {
        'display_name': 'Leaf Smut (Entyloma oryzae)',
        'symptoms': 'Small, slightly raised, angular black spots on both leaf surfaces, turning gray with age.',
        'causes': 'Airborne fungal spores infecting mature leaves in late growth stages.',
        'treatment': '1. Foliar spray of Propiconazole 25 EC or Mancozeb.\n2. Burn or plow under diseased crop residues after harvest.'
    }
}


def find_default_model():
    """Finds the most recently saved model in saved_models directory."""
    if not os.path.exists(SAVED_MODELS_DIR):
        return None
    files = [os.path.join(SAVED_MODELS_DIR, f) for f in os.listdir(SAVED_MODELS_DIR) if f.endswith('.joblib')]
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def predict_disease(image_path: str, model_path: str = None) -> dict:
    """
    Predicts the rice crop disease for an input image.

    Args:
        image_path: Path to the rice leaf image.
        model_path: Path to the serialized .joblib model bundle.

    Returns:
        Dictionary containing prediction results, probabilities, and treatment advisory.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    # Load model
    if model_path is None or not os.path.exists(model_path):
        model_path = find_default_model()
        if model_path is None:
            raise FileNotFoundError(
                "No trained model found in 'saved_models/'. Please train a model first using: python train.py"
            )

    model_data = joblib.load(model_path)
    pipeline = model_data['pipeline']
    class_names = model_data['class_names']
    levels = model_data.get('levels', ML_LEVELS)

    # 1. Preprocess: resize to 224x224 and apply logarithmic enhancement (c=2.0)
    preprocessed_img = preprocess_image(image_path, enhance=True, augment=False)

    # 2. Extract fused features (ML-MCLBP + Color Correlogram)
    features = extract_fused_features(preprocessed_img, level=levels)
    features_2d = features.reshape(1, -1)

    # 3. Predict class
    pred_idx = pipeline.predict(features_2d)[0]
    predicted_class = class_names[pred_idx]

    # Probabilities if supported
    probabilities = {}
    confidence = 0.0
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(features_2d)[0]
        confidence = float(probs[pred_idx]) * 100.0
        probabilities = {class_names[i]: float(probs[i]) * 100.0 for i in range(len(class_names))}
    else:
        confidence = 100.0
        probabilities = {predicted_class: 100.0}

    # Normalize class key for advisory lookup
    clean_key = predicted_class.lower().replace(" ", "").replace("_", "")
    advisory = DISEASE_ADVISORY.get(clean_key, {
        'display_name': predicted_class,
        'symptoms': 'Consult local agricultural extension office for symptom verification.',
        'causes': 'Phytopathogen infection.',
        'treatment': 'Standard broad-spectrum fungicide/bactericide application.'
    })

    return {
        'image_path': image_path,
        'predicted_class': predicted_class,
        'confidence (%)': confidence,
        'probabilities': probabilities,
        'advisory': advisory,
        'model_name': model_data.get('model_name', 'Linear SVM'),
        'dataset': model_data.get('dataset', 'Unknown')
    }


def main():
    parser = argparse.ArgumentParser(description="Predict Disease in Rice Leaf Image")
    parser.add_argument('--image', type=str, required=True, help="Path to input rice leaf image")
    parser.add_argument('--model', type=str, default=None, help="Path to trained .joblib model")
    args = parser.parse_args()

    try:
        res = predict_disease(args.image, args.model)
        adv = res['advisory']

        print("\n" + "=" * 70)
        print("          RICE LEAF DISEASE DIAGNOSIS REPORT")
        print("=" * 70)
        print(f"Input Image:   {res['image_path']}")
        print(f"Model Used:    {res['model_name']} (Trained on {res['dataset']})")
        print(f"Diagnosis:     {adv['display_name']}")
        print(f"Confidence:    {res['confidence (%)']:.2f}%")
        print("-" * 70)
        print("Class Probabilities:")
        for cls, prob in sorted(res['probabilities'].items(), key=lambda x: x[1], reverse=True):
            bar = "#" * int(prob / 5)
            print(f"  {cls:<18} : {prob:>6.2f}%  {bar}")
        print("-" * 70)
        print(f"Symptoms:\n  {adv['symptoms']}\n")
        print(f"Causes:\n  {adv['causes']}\n")
        print(f"Recommended Agronomic Treatment:\n  {adv['treatment']}")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
