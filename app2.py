import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from cvzone.HandTrackingModule import HandDetector
import pandas as pd
import random

# -------------------- Basic Page Setup --------------------
st.set_page_config(page_title="ASL Crossword", layout="wide")

def set_bg_color():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: #90021f;  
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <style>
    [data-testid="stHeader"] {
        background-color: #131720 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧩 ASL Crossword Puzzle Game")

set_bg_color()

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, 36)
    model.load_state_dict(torch.load("asl_mobilenetv2_best.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

labels = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------- Video Transformer --------------------
class ASLTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = HandDetector(maxHands=1)
        self.current_sign = ""
        self.current_probs = None
        self.input_frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        hands, img = self.detector.findHands(img)

        if hands:
            x, y, w, h = hands[0]['bbox']
            offset = 20
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
                imgGray = cv2.equalizeHist(imgGray)
                pil_img = Image.fromarray(imgGray)
                img_tensor = transform(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(img_tensor)
                    probs = F.softmax(output, dim=1)
                    confidence, predicted = torch.max(probs, 1)
                    sign = labels[predicted.item()]
                    conf = confidence.item()

                self.current_sign = sign
                self.current_probs = probs.cpu()
                self.input_frame = imgGray

                label_y = y - 10 if y - 30 > 10 else y + h + 30
                cv2.putText(
                    img,
                    f"{sign} ({conf:.2f})",
                    (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return img

# -------------------- Load Crossword Data --------------------
def load_crossword_data():
    crossword_data = pd.read_csv("kids_crossword_data.csv")
    return crossword_data

# -------------------- Create Crossword Hints --------------------
def create_clues(crossword_data):
    across_clues = []
    down_clues = []

    for _, row in crossword_data.iterrows():
        clue = row['clue']
        orientation = row['orientation']
        
        if orientation == 'across':
            across_clues.append(f"Clue: {clue}")
        else:
            down_clues.append(f"Clue: {clue}")

    return across_clues, down_clues

# -------------------- Crossword Board Setup --------------------
rows, cols = 5, 5

if "board" not in st.session_state:
    st.session_state.board = [["" for _ in range(cols)] for _ in range(rows)]

if "selected_cell" not in st.session_state:
    st.session_state.selected_cell = None

if "current_letter" not in st.session_state:
    st.session_state.current_letter = ""

st.header("📋 Crossword Board")

def get_cell_style(selected):
    if selected:
        return (
            "background: linear-gradient(135deg, #FFD700, #FFA500); "
            "color: black; font-size: 24px; font-weight: bold; border: 2px solid #000000; "
            "border-radius: 12px; height: 70px; width: 100%; text-align: center; vertical-align: middle;"
            "box-shadow: 0 0 10px 3px rgba(255, 215, 0, 0.7); transition: 0.3s ease;"
        )
    else:
        return (
            "background: linear-gradient(135deg, #FFB6C1, #FF69B4); "
            "color: white; font-size: 24px; font-weight: bold; border: 2px solid #ff69b4; "
            "border-radius: 12px; height: 70px; width: 100%; text-align: center; vertical-align: middle;"
            "transition: 0.3s ease;"
        )

for row_idx in range(rows):
    col_objs = st.columns(cols)
    for col_idx in range(cols):
        cell_value = st.session_state.board[row_idx][col_idx]
        selected = (st.session_state.selected_cell == (row_idx, col_idx))
        cell_content = cell_value if cell_value else " "

        with col_objs[col_idx]:
            if st.button(cell_content, key=f"btn-{row_idx}-{col_idx}", use_container_width=True):
                st.session_state.selected_cell = (row_idx, col_idx)

            st.markdown(
                f"""
                <style>
                div[data-testid="stButton"][key="btn-{row_idx}-{col_idx}"] > button {{
                    {get_cell_style(selected)}
                }}
                div[data-testid="stButton"][key="btn-{row_idx}-{col_idx}"]:hover > button {{
                    background: linear-gradient(135deg, #87CEFA, #4682B4);
                    border: 2px solid #4682B4;
                    color: white;
                    transform: scale(1.05);
                }}
                </style>
                """,
                unsafe_allow_html=True
            )

# -------------------- Webcam & Prediction --------------------
if st.session_state.selected_cell:
    st.success(f"Selected Cell: {st.session_state.selected_cell}")
    webrtc_ctx = webrtc_streamer(
        key=f"asl-crossword-{st.session_state.selected_cell}",
        video_transformer_factory=ASLTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.video_transformer:
        predicted_sign = webrtc_ctx.video_transformer.current_sign
        probs = webrtc_ctx.video_transformer.current_probs
        input_frame = webrtc_ctx.video_transformer.input_frame

        if predicted_sign:
            st.session_state.current_letter = predicted_sign
            st.info(f"Detected Letter: {predicted_sign}")

        # ✅ Visualization: Input Image
        if input_frame is not None:
            st.subheader("🖼️ Input Sent to Model")
            st.image(input_frame, caption="Preprocessed Grayscale Hand Image", channels="GRAY")

        # ✅ Visualization: Top 5 Predictions Bar Chart
        if probs is not None:
            st.subheader("📊 Top 5 Model Predictions")
            top_probs, top_indices = torch.topk(probs.squeeze(), 5)
            top_labels = [labels[i] for i in top_indices.tolist()]
            prob_df = pd.DataFrame({'Sign': top_labels, 'Confidence': top_probs.tolist()})
            st.bar_chart(prob_df.set_index('Sign'))

    if st.button("✅ Confirm Letter"):
        row, col = st.session_state.selected_cell
        st.session_state.board[row][col] = st.session_state.current_letter
        st.session_state.selected_cell = None
        st.session_state.current_letter = ""

# -------------------- Hints and Chart --------------------
crossword_data = load_crossword_data()
across_clues, down_clues = create_clues(crossword_data)

col1, col2 = st.columns([0.65, 0.35])

with col1:
    st.header("🧩 Crossword Hints")

    across_col, down_col = st.columns(2)

    with across_col:
        st.markdown("<h3>Across:</h3>", unsafe_allow_html=True)
        st.markdown("<ul style='font-size:20px;'>", unsafe_allow_html=True)
        for clue in across_clues:
            st.markdown(f"<li>{clue}</li>", unsafe_allow_html=True)
        st.markdown("</ul>", unsafe_allow_html=True)

    with down_col:
        st.markdown("<h3>Down:</h3>", unsafe_allow_html=True)
        st.markdown("<ul style='font-size:20px;'>", unsafe_allow_html=True)
        for clue in down_clues:
            st.markdown(f"<li>{clue}</li>", unsafe_allow_html=True)
        st.markdown("</ul>", unsafe_allow_html=True)

with col2:
    st.subheader("🤟 ASL Alphabet Reference")
    asl_chart = Image.open("asl_alphabets.jpg")
    st.image(asl_chart, caption="American Sign Language Alphabets", use_container_width=True)

    st.markdown(
        """
        <p style='font-size:18px; text-align:center; color:gray;'>
            Use this reference to make correct hand signs!
        </p>
        """, unsafe_allow_html=True
    )

st.markdown("---")
st.caption("Tip: Make clear gestures! ✋🏻 Good lighting helps recognition. 🚀")
