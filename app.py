
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics.pairwise import cosine_similarity

# 기본 설정
GALLERY_DIR = "./gallery"
st.set_page_config(page_title="내가 그린 은하는 어떤 은하?", layout="centered")
st.title("🎨 내가 그린 은하를 분석해볼까요? (CNN 기반 유사도 검색)")

# canvas 인터페이스
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)", 
    stroke_width=6,
    stroke_color=st.color_picker("🖌 선 색상 선택", "#ffffff"),
    background_color="#000000",
    width=256,
    height=256,
    drawing_mode="freedraw",
    key="canvas",
)

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# CNN 특성 추출기
@st.cache_resource
def load_model():
    resnet = models.resnet18(pretrained=True)
    model = nn.Sequential(*list(resnet.children())[:-1])
    model.eval()
    return model

model = load_model()

# gallery 이미지 feature vector 미리 추출
@st.cache_resource
def extract_gallery_features():
    vectors = []
    paths = []
    for fname in os.listdir(GALLERY_DIR):
        if fname.endswith(('.jpg', '.png')):
            path = os.path.join(GALLERY_DIR, fname)
            try:
                img = Image.open(path).convert('RGB')
                tensor = transform(img).unsqueeze(0)
                feat = model(tensor).view(-1).detach().numpy()
                vectors.append(feat)
                paths.append(path)
            except:
                continue
    return vectors, paths

gallery_features, gallery_paths = extract_gallery_features()

# 유사도 계산 함수
def find_similar_images(user_img, top_k=3):
    tensor = transform(user_img).unsqueeze(0)
    user_feat = model(tensor).view(-1).detach().numpy().reshape(1, -1)

    sims = cosine_similarity(user_feat, gallery_features)[0]
    top_indices = sims.argsort()[::-1][:top_k]
    return [(gallery_paths[i], sims[i]) for i in top_indices]

# 분석 버튼
if st.button("✨ CNN 기반 분석 시작"):
    if canvas_result.image_data is not None:
        user_img = Image.fromarray(np.uint8(canvas_result.image_data)).convert("RGB")
        st.image(user_img, caption="🖼️ 당신이 그린 은하", width=256)

        with st.spinner("비슷한 은하를 찾는 중..."):
            results = find_similar_images(user_img)

        st.markdown("### 🔍 유사 은하 Top 3")
        for path, score in results:
            st.image(path, caption=f"유사도: {score:.4f}", width=240)
    else:
        st.warning("먼저 은하를 그려주세요!")
