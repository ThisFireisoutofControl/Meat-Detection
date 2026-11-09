import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import time
import av
import os
import urllib.request
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Smart BBQ Tracker", layout="wide")
st.title("🥩 Real-Time Meat Doneness & Sear Tracker")

# ----------------- 1. โหลดโมเดล -----------------
MODEL_PATH = "best.pt"
# ใส่ URL ของ GitHub Releases หรือลิงก์ตรงของคุณ
MODEL_URL = "https://github.com/ThisFireisoutofControl/Meat-Detection/releases/download/v1.0.0/best.pt"

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        with st.spinner("กำลังดาวน์โหลดไฟล์โมเดล... กรุณารอสักครู่"):
            headers = {"User-Agent": "Mozilla/5.0"}
            req = urllib.request.Request(MODEL_URL, headers=headers)
            with urllib.request.urlopen(req) as response, open(MODEL_PATH, "wb") as out_file:
                out_file.write(response.read())
    return YOLO(MODEL_PATH)

try:
    model = get_model()
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
    st.stop()

# ----------------- 2. การตั้งค่า Sidebar -----------------
st.sidebar.title("⚙️ การตั้งค่าโหมดการทำงาน")
app_mode = st.sidebar.radio(
    "เลือกแหล่งข้อมูลที่ต้องการตรวจจับ:",
    ["📷 กล้องสด & Real-Time Tracking (WebRTC)", "🖼️ อัปโหลดภาพนิ่งเอง (Upload Image)", "📸 ถ่ายภาพหน้าจอ/กล้อง (Snapshot)"]
)

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.35, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold (NMS)", 0.1, 1.0, 0.45, 0.05)

# ฟังก์ชันประมวลผลสำหรับภาพนิ่ง
def process_static_image(image_bgr):
    results = model.predict(image_bgr, conf=conf_threshold, iou=iou_threshold)
    result = results[0]
    
    # นับจำนวนแต่ละคลาส
    counts = {"meat_raw": 0, "meat_cooked": 0, "meat_burnt": 0}
    if result.boxes is not None:
        for cls_id in result.boxes.cls.int().cpu().tolist():
            class_name = model.names[cls_id]
            counts[class_name] = counts.get(class_name, 0) + 1
            
    annotated_img = result.plot()
    # แปลงสีกลับเป็น RGB เพื่อแสดงผลบน Streamlit
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    return annotated_img_rgb, counts

# ----------------- 3. แยกการทำงานตามโหมด -----------------

# โหมดที่ 1: กล้องสด WebRTC + Tracking
if app_mode == "📷 กล้องสด & Real-Time Tracking (WebRTC)":
    st.subheader("ระบบติดตามชิ้นเนื้อและนับเวลาถอยหลังแบบเรียลไทม์")
    
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    class MeatTrackerTransformer(VideoTransformerBase):
        def __init__(self):
            self.meat_records = {}
            self.cooked_timeout = 30.0

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            current_time = time.time()

            results = model.track(img, persist=True, tracker="bytetrack.yaml", verbose=False, conf=conf_threshold, iou=iou_threshold)
            result = results[0]

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()
                classes = result.boxes.cls.int().cpu().tolist()

                for box, track_id, cls_id in zip(boxes, track_ids, classes):
                    x1, y1, x2, y2 = map(int, box)
                    cls_name = model.names[cls_id]

                    if track_id not in self.meat_records:
                        self.meat_records[track_id] = {
                            "first_seen": current_time,
                            "cooked_start": None,
                            "status": cls_name
                        }

                    record = self.meat_records[track_id]
                    record["status"] = cls_name

                    box_color = (0, 165, 255)
                    label_text = f"ID:{track_id} {cls_name}"

                    if cls_name == "meat_raw":
                        total_grill_time = int(current_time - record["first_seen"])
                        label_text += f" ({total_grill_time}s)"
                        box_color = (180, 105, 255)
                    elif cls_name == "meat_cooked":
                        if record["cooked_start"] is None:
                            record["cooked_start"] = current_time
                        cooked_duration = current_time - record["cooked_start"]
                        time_left = max(0, int(self.cooked_timeout - cooked_duration))
                        if time_left > 0:
                            box_color = (0, 255, 0)
                            label_text = f"ID:{track_id} EAT NOW! ({time_left}s)"
                        else:
                            box_color = (0, 0, 255)
                            label_text = f"ID:{track_id} OVERCOOKED!"
                    elif cls_name == "meat_burnt":
                        box_color = (0, 0, 255)
                        label_text = f"ID:{track_id} BURNT!"

                    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
                    (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img, (x1, y1 - 25), (x1 + w, y1), box_color, -1)
                    cv2.putText(img, label_text, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="meat-tracker",
        video_transformer_factory=MeatTrackerTransformer,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False}
    )

# โหมดที่ 2: อัปโหลดภาพเข้ามาเอง
elif app_mode == "🖼️ อัปโหลดภาพนิ่งเอง (Upload Image)":
    st.subheader("อัปโหลดไฟล์ภาพเนื้อย่างเพื่อวิเคราะห์")
    uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพ...", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        
        col1, col2 = st.columns([2, 1])
        with st.spinner("กำลังประมวลผล..."):
            annotated_img, counts = process_static_image(img_bgr)
            
        with col1:
            st.image(annotated_img, use_container_width=True)
        with col2:
            st.markdown("### 📊 สรุปจำนวนชิ้นเนื้อ")
            st.metric("🥩 ดิบ (Raw)", f"{counts.get('meat_raw', 0)} ชิ้น")
            st.metric("🍖 สุกพร้อมทาน (Cooked)", f"{counts.get('meat_cooked', 0)} ชิ้น")
            st.metric("🔥 ไหม้ (Burnt)", f"{counts.get('meat_burnt', 0)} ชิ้น")
            
            if counts.get('meat_burnt', 0) > 0:
                st.error("⚠️ ตรวจพบเนื้อไหม้บนเตา!")
            elif counts.get('meat_cooked', 0) > 0:
                st.success("✅ มีเนื้อสุกพร้อมรับประทานแล้ว")

# โหมดที่ 3: ถ่ายภาพสดจากกล้อง / หน้าจอ (Snapshot)
elif app_mode == "📸 ถ่ายภาพหน้าจอ/กล้อง (Snapshot)":
    st.subheader("กดถ่ายภาพจากกล้องเว็บแคมหรือภาพหน้าจอ")
    camera_file = st.camera_input("กดปุ่ม Take Photo เพื่อส่งภาพเข้าโมเดล")
    
    if camera_file is not None:
        file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        
        col1, col2 = st.columns([2, 1])
        with st.spinner("กำลังประมวลผล..."):
            annotated_img, counts = process_static_image(img_bgr)
            
        with col1:
            st.image(annotated_img, use_container_width=True)
        with col2:
            st.markdown("### 📊 สรุปจำนวนชิ้นเนื้อ")
            st.metric("🥩 ดิบ (Raw)", f"{counts.get('meat_raw', 0)} ชิ้น")
            st.metric("🍖 สุกพร้อมทาน (Cooked)", f"{counts.get('meat_cooked', 0)} ชิ้น")
            st.metric("🔥 ไหม้ (Burnt)", f"{counts.get('meat_burnt', 0)} ชิ้น")
                            
