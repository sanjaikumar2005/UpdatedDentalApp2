import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import speech_recognition as sr
import requests
import os

# =============================
# CONFIG
# =============================
MODEL_URL = "https://drive.google.com/uc?id=1jFsvVVLK_VBtGiRcHj-Hv0cBOs-FjBCu"
MODEL_PATH = "model.pt"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading AI model (one-time)..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

CLASS_NAMES = ["Calculus", "Gingivitis"]


HF_API_KEY = "hf_zEEAapJUSQTNPOlWdyBuhYVlDedyjR"
API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# =============================
# OFFLINE DENTAL KNOWLEDGE BASE
# =============================
OFFLINE_ANSWERS = {
    "pain": "🦷 Tooth pain may be due to cavity or infection.\nTamil: பல் வலி தொற்று அல்லது கேவிட்டி காரணமாக.",
    "toothache": "🦷 Toothache indicates nerve involvement.\nTamil: நரம்பு பாதிப்பு இருக்கலாம்.",
    "bleeding": "🩸 Bleeding gums are sign of gingivitis.\nTamil: ஈறு இரத்தம் – ஜிஞ்சிவைட்டிஸ்.",
    "swollen": "🤕 Gum swelling indicates inflammation.\nTamil: ஈறு வீக்கம்.",
    "gingivitis": "🦷 Gingivitis is early gum disease.\nTamil: ஆரம்ப ஈறு நோய்.",
    "calculus": "🪨 Calculus is hardened plaque.\nTamil: உறைந்த பல் கல்.",
    "tartar": "🪨 Tartar requires scaling.\nTamil: ஸ்கேலிங் தேவை.",
    "bad breath": "😷 Bad breath caused by bacteria.\nTamil: வாய் துர்நாற்றம்.",
    "mouth smell": "😷 Poor oral hygiene.\nTamil: வாய்சுத்தம் குறைவு.",
    "cavity": "🕳️ Tooth decay present.\nTamil: பல் அழுகல்.",
    "hole": "🕳️ Hole indicates cavity.\nTamil: கேவிட்டி.",
    "pus": "⚠️ Pus indicates infection.\nTamil: தீவிர தொற்று.",
    "abscess": "🚨 Dental abscess emergency.\nTamil: அவசர நிலை.",
    "loose": "⚠️ Loose tooth due to gum disease.\nTamil: ஈறு நோய்.",
    "ulcer": "😖 Mouth ulcer heals in days.\nTamil: வாய்ப்புண்.",
    "sensitivity": "❄️ Sensitivity due to enamel loss.\nTamil: பல் பாதுகாப்பு குறைவு.",
    "healthy": "✅ Teeth appear healthy.\nTamil: பற்கள் ஆரோக்கியம்."
}

# =============================
# FUNCTIONS
# =============================
def offline_answer(question):
    question = question.lower()
    matches = []
    for key, ans in OFFLINE_ANSWERS.items():
        if key in question:
            matches.append(ans)

    if matches:
        return "\n\n".join(matches)
    else:
        return "🦷 Please consult a dentist.\nTamil: மருத்துவரை அணுகவும்."

def ai_answer(question):
    prompt = f"""
You are a dental doctor.
Explain simply for patients.
Answer in Tamil + English.

Question: {question}
Answer:
"""
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200}}
    response = requests.post(API_URL, headers=headers, json=payload, timeout=15)

    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list):
            return result[0]["generated_text"]
    raise Exception("AI Busy")

@st.cache_resource
def load_model():
    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    return model

# =============================
# LOAD MODEL
# =============================
model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =============================
# UI
# =============================
st.set_page_config(page_title="Smart Dental AI", page_icon="🦷")
st.title("🦷 Smart Dental Diagnosis & Assistant")

# =============================
# IMAGE INPUT
# =============================
st.subheader("📷 Upload Image / Camera")
img = st.camera_input("Camera") or st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if img:
    image = Image.open(img).convert("RGB")
    st.image(image, use_column_width=True)

    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1).numpy()[0]

    pred = np.argmax(probs)
    confidence = probs[pred] * 100
    disease = CLASS_NAMES[pred]

    st.success(f"🧠 Prediction: **{disease}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")

    # =============================
    # CURE SUGGESTIONS
    # =============================
    st.subheader("💊 Patient Care Advice")

    if disease == "Calculus":
        st.write("""
        • Professional scaling required  
        • Brush twice daily  
        • Use anti-plaque mouthwash  
        • Avoid tobacco  
        """)
    elif disease == "Gingivitis":
        st.write("""
        • Maintain oral hygiene  
        • Use medicated mouthwash  
        • Avoid sugary food  
        • Visit dentist if bleeding continues  
        """)
    else:
        st.write("""
        • Teeth look healthy  
        • Continue brushing twice daily  
        • Regular dental checkups  
        """)

# =============================
# QUESTION SECTION
# =============================
st.divider()
st.subheader("💬 Ask Dental Question")

text_q = st.text_input("Type your question")

# =============================
# AI + FALLBACK LOGIC
# =============================
if text_q:
    with st.spinner("Thinking..."):
        try:
            answer = ai_answer(text_q)
            st.success("🤖 AI Answer")
            st.write(answer)
        except:
            st.warning("⚠️ AI busy – showing doctor knowledge")
            st.info(offline_answer(text_q))

st.caption("⚕️ Educational use only – consult dentist for treatment")
