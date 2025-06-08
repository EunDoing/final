import os
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_drawable_canvas import st_canvas
import numpy as np

# ✅ 이미지 폴더 설정
IMG_DIR = 'downloaded_images_100'
os.makedirs(IMG_DIR, exist_ok=True)

# ✅ CNN 특성 추출기 정의
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            return x.view(x.size(0), -1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FeatureExtractor().to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ 갤러리 이미지 불러오기 + 특징 추출
@st.cache_data
def load_gallery_features():
    paths = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
    features, valid_paths = [], []
    for path in paths:
        try:
            img = Image.open(path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            feat = model(tensor).cpu().numpy()[0]
            features.append(feat)
            valid_paths.append(path)
        except:
            continue
    return valid_paths, features

# ✅ 유사 이미지 검색
def find_similar_images(user_img, paths, vectors, top_k=3):
    user_tensor = transform(user_img.convert('RGB')).unsqueeze(0).to(device)
    user_feat = model(user_tensor).cpu().numpy()
    sims = cosine_similarity(user_feat, vectors)[0]
    top = sims.argsort()[::-1][:top_k]
    return [(paths[i], sims[i]) for i in top]

# ✅ Streamlit UI
st.set_page_config(page_title="은하 창조", layout="centered")
st.title("🎨 내가 만든 은하가 이미 존재한다고?")
st.markdown("🪐 아래 캔버스에 은하를 그리면, 실제 SDSS 은하 중 유사한 이미지를 찾아줘요!")

stroke_color = st.color_picker(" 선 색상 선택", "#ffffff"

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=6,
    stroke_color=stroke_color,
    background_color="#1a1a3d",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data).astype(np.uint8)).convert('RGB')
    st.image(img, caption="내가 그린 은하", width=200)

    if st.button("🔍 유사한 은하 찾기"):
        gallery_paths, gallery_vectors = load_gallery_features()
        results = find_similar_images(img, gallery_paths, gallery_vectors)
        st.subheader("🔎 가장 유사한 은하 이미지")
        for path, score in results:
            st.image(path, caption=f"유사도: {score:.4f}", use_container_width=True)
