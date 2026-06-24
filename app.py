import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# 1. Page Configuration & Title
st.set_page_config(page_title="Live TACO Trash Detector", layout="centered")
st.title("🗑️ Live TACO Trash Detector")
st.write("Point your camera at waste items to scan your environment in real time.")

# 2. Connect to Roboflow Client
@st.cache_resource
def get_inference_client():
    return InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="zs2Fpdr6GYFftsJ7kQws"
    )

client = get_inference_client()

# 3. Define the Real-Time Frame Processor
class TrashDetectorTransformer(VideoTransformerBase):
    def __init__(self):
        # Fallback fonts configuration
        try:
            self.font_big = ImageFont.truetype("arial.ttf", size=18)
        except IOError:
            try:
                self.font_big = ImageFont.truetype("DejaVuSans.ttf", size=18)
            except IOError:
                self.font_big = ImageFont.load_default()

    def transform(self, frame):
        # Convert incoming WebRTC frame to OpenCV BGR image format
        img_bgr = frame.to_ndarray(format="bgr24")
        
        # Convert to RGB for PIL processing and the SDK requirements
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        try:
            # 4. Call your Roboflow workflow framework on the raw frame
            result = client.run_workflow(
                workspace_name="dylans-workspace-gchst",
                workflow_id="trash-detection-workflow-1782289148660",
                images={"image": pil_img},
                parameters={"classes": "paper, plastic, glass, metal, cardboard"},
                use_cache=True  
            )
            
            # Parse Response Data blocks
            workflow_output = result[0] if isinstance(result, list) else result
            predictions_data = workflow_output.get("predictions", {})
            detected_items = predictions_data.get("predictions", [])
            
            # 5. Draw Annotations on top of the live frame
            if detected_items:
                draw = ImageDraw.Draw(pil_img)
                img_width, img_height = pil_img.size
                box_width = max(2, int(img_width * 0.005))
                
                for item in detected_items:
                    w = item.get('width')
                    h = item.get('height')
                    x0 = item.get('x') - (w / 2)
                    y0 = item.get('y') - (h / 2)
                    x1 = item.get('x') + (w / 2)
                    y1 = item.get('y') + (h / 2)
                    
                    label = f"{item.get('class')} ({item.get('confidence'):.0%})"
                    
                    # Drawing elements
                    draw.rectangle([x0, y0, x1, y1], outline="#00FF00", width=box_width)
                    try:
                        text_w, text_h = draw.textbbox((0, 0), label, font=self.font_big)[2:]
                    except AttributeError:
                        text_w, text_h = draw.textsize(label, font=self.font_big)
                        
                    draw.rectangle([x0, y0 - text_h - 6, x0 + text_w + 10, y0], fill="#00FF00")
                    draw.text((x0 + 5, y0 - text_h - 4), label, fill="#000000", font=self.font_big)
                    
            # Transform annotated picture framework back into standard OpenCV BGR structure
            final_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return final_bgr
            
        except Exception:
            # If API limits hit or connection drops momentarily, bypass gracefully to keep loop running
            return img_bgr

# 4. Run Live Stream Component
webrtc_streamer(
    key="trash-detector", 
    video_transformer_factory=TrashDetectorTransformer,
    media_stream_constraints={"video": True, "audio": False}
)