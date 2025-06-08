
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import os
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

# 설정
GALLERY_DIR = "./gallery"
st.set_page_config(page_title="내가 그린 은하는 어떤 은하?", layout="centered")
st.title("🎨 내가 그린 은하를 분석해볼까요?")

# 캔버스 설정
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
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def find_top_similar_images(input_pil, gallery_dir=GALLERY_DIR, top_n=3):
    input_tensor = transform(input_pil).view(-1).numpy().reshape(1, -1)

    similarities = []
    paths = []

    for fname in os.listdir(gallery_dir):
        if fname.endswith(('.jpg', '.png')):
            path = os.path.join(gallery_dir, fname)
            gallery_img = Image.open(path).convert("RGB")
            gallery_tensor = transform(gallery_img).view(-1).numpy().reshape(1, -1)

            sim = cosine_similarity(input_tensor, gallery_tensor)[0][0]
            similarities.append(sim)
            paths.append(path)

    sorted_idx = np.argsort(similarities)[::-1][:top_n]
    top_paths = [paths[i] for i in sorted_idx]
    top_scores = [similarities[i] for i in sorted_idx]
    return top_paths, top_scores

# 실행 버튼
if st.button("✨ 분석 시작"):
    if canvas_result.image_data is not None:
        # canvas → PIL
        user_img = Image.fromarray(np.uint8(canvas_result.image_data)).convert("RGB")

        st.image(user_img, caption="🖼️ 당신이 그린 은하", width=256)

        with st.spinner("분석 중..."):
            top_paths, top_scores = find_top_similar_images(user_img)

        st.markdown("### 🔍 가장 유사한 은하들")
        for i in range(len(top_paths)):
            st.image(top_paths[i], caption=f"유사도: {top_scores[i]:.4f}", width=240)
    else:
        st.warning("먼저 은하를 그려주세요!")
