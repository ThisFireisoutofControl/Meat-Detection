import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import time
import av
import os
import gdown
from ultralytics import YOLO

st.set_page_config(page_title="Smart BBQ Tracker", layout="wide")
st.title("🥩 Real-Time Individual Meat Tracker & Timer")

# กำหนด WebRTC Configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- [จุดที่แก้ไข: ตรวจสอบและดาวน์โหลดโมเดลอัตโนมัติ] ---
MODEL_PATH = "best.pt"
# ตรวจสอบให้แน่ใจว่าใส่เฉพาะ ID (ตัวเลขและตัวอักษรยาวๆ)
GDRIVE_FILE_ID = "https://drive.google.com/file/d/1iVZoYRtaRmeiKOgTjnQjy4vv_513sJjl/view?usp=sharing"

@st.cache_resource
def get_model():
    # ถ้าไฟล์ยังไม่มี หรือขนาดเล็กผิดปกติ (น้อยกว่า 1MB แปลว่าเป็นไฟล์ HTML error)
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)  # ลบไฟล์เสียเดิมทิ้งก่อน
            
        with st.spinner("กำลังดาวน์โหลดไฟล์โมเดล... กรุณารอสักครู่"):
            # ใช้ gdown ด้วย id ตรงๆ พร้อม fuzzy=True เพื่อข้ามหน้ายืนยันไวรัส
            gdown.download(id=GDRIVE_FILE_ID, output=MODEL_PATH, quiet=False, fuzzy=True)
            
    return YOLO(MODEL_PATH)

model = get_model()

# คลาสประมวลผลวิดีโอแบบเรียลไทม์ผ่าน WebRTC
class MeatTrackerTransformer(VideoTransformerBase):
    def __init__(self):
        # โครงสร้างเก็บข้อมูล: {track_id: {"first_seen": timestamp, "cooked_start": timestamp, "status": str}}
        self.meat_records = {}
        self.cooked_timeout = 30.0  # เตือนถ้าสุกเกิน 30 วินาทีแล้วยังไม่คีบ

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()

        # รัน Tracking แบบ Real-time ด้วย ByteTrack
        results = model.track(img, persist=True, tracker="bytetrack.yaml", verbose=False)
        result = results[0]

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()
            classes = result.boxes.cls.int().cpu().tolist()

            for box, track_id, cls_id in zip(boxes, track_ids, classes):
                x1, y1, x2, y2 = map(int, box)
                cls_name = model.names[cls_id]

                # 1. บันทึก ID ใหม่ที่เพิ่งตรวจพบ
                if track_id not in self.meat_records:
                    self.meat_records[track_id] = {
                        "first_seen": current_time,
                        "cooked_start": None,
                        "status": cls_name
                    }

                record = self.meat_records[track_id]
                record["status"] = cls_name

                # 2. จัดการตัวจับเวลาตามระดับความสุก
                box_color = (0, 165, 255)  # สีส้มเริ่มต้น (Raw)
                label_text = f"ID:{track_id} {cls_name}"

                if cls_name == "meat_raw":
                    total_grill_time = int(current_time - record["first_seen"])
                    label_text += f" ({total_grill_time}s)"
                    box_color = (180, 105, 255)  # ชมพูอมม่วง

                elif cls_name == "meat_cooked":
                    # เริ่มนับเวลาตอนที่สุก
                    if record["cooked_start"] is None:
                        record["cooked_start"] = current_time

                    cooked_duration = current_time - record["cooked_start"]
                    time_left = max(0, int(self.cooked_timeout - cooked_duration))
                    
                    if time_left > 0:
                        box_color = (0, 255, 0)  # สีเขียว: พร้อมทาน
                        label_text = f"ID:{track_id} EAT NOW! ({time_left}s)"
                    else:
                        box_color = (0, 0, 255)  # สุกนานเกินไป เสี่ยงไหม้
                        label_text = f"ID:{track_id} OVERCOOKED!"

                elif cls_name == "meat_burnt":
                    box_color = (0, 0, 255)  # สีแดง
                    label_text = f"ID:{track_id} BURNT! REMOVE"

                # 3. วาด Bounding Box และป้ายกำกับ
                cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
                
                # วาดพื้นหลังป้ายข้อความให้อ่านง่าย
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, y1 - 25), (x1 + w, y1), box_color, -1)
                cv2.putText(img, label_text, (x1, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# สตรีมวิดีโอผ่านหน้าเว็บ
webrtc_streamer(
    key="meat-tracker",
    video_transformer_factory=MeatTrackerTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)
