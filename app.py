import streamlit as st

from inference_sdk import InferenceHTTPClient

from PIL import Image, ImageDraw, ImageFont

import os



# 1. Page Configuration & Title

st.set_page_config(page_title="TACO Trash Detection Tester", layout="centered")

st.title("🗑️ TACO Trash Detection Tester")

st.write("Scan your environment using live camera input or upload an image to detect trash items.")



# 2. Connect to Roboflow Client

@st.cache_resource

def get_inference_client():

    return InferenceHTTPClient(

        api_url="https://serverless.roboflow.com",

        api_key="zs2Fpdr6GYFftsJ7kQws"

    )



client = get_inference_client()



# 3. Choose Input Method

input_method = st.radio("Select Input Method:", ["📷 Live Camera", "📁 Upload Image"])



source_file = None



if input_method == "📷 Live Camera":

    # Streamlit's built-in camera widget (works on phones and laptops)

    source_file = st.camera_input("Take a snapshot of trash")

else:

    # Classic file uploader fallback

    source_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])



if source_file is not None:

    # Load image via PIL

    original_image = Image.open(source_file).convert("RGB")

   

    # Save the file temporarily for the SDK to read

    temp_path = "temp_testing_image.jpg"

    original_image.save(temp_path)

   

    # Button to run inference

    if st.button("Run Trash Detection"):

        with st.spinner("Analyzing image and drawing bounding boxes..."):

            try:

                # 4. Run your workflow

                result = client.run_workflow(

                    workspace_name="dylans-workspace-gchst",

                    workflow_id="trash-detection-workflow-1782289148660",

                    images={"image": temp_path},

                    parameters={"classes": "paper, plastic, glass, metal, cardboard"},

                    use_cache=True  

                )

               

                # Parse Response Data

                workflow_output = result[0] if isinstance(result, list) else result

                predictions_data = workflow_output.get("predictions", {})

                detected_items = predictions_data.get("predictions", [])

                trash_count = workflow_output.get('trash_count', 0)

               

                st.success("Analysis Complete!")

                st.metric(label="Total Trash Items Counted", value=trash_count)

               

                if detected_items:

                    # 5. Draw Bounding Boxes onto a copy of the image

                    annotated_image = original_image.copy()

                    draw = ImageDraw.Draw(annotated_image)

                   

                    # 🌟 DYNAMIC SCALING SETUP 🌟

                    img_width, img_height = annotated_image.size

                   

                    # Compute box thickness and font size based on image width

                    box_width = max(2, int(img_width * 0.005))       # 0.5% of image width

                    dynamic_size = max(12, int(img_width * 0.025))   # 2.5% of image width

                   

                    try:

                        font = ImageFont.truetype("arial.ttf", size=dynamic_size)

                    except IOError:

                        # Fallback for systems without Arial installed

                        try:

                            font = ImageFont.truetype("DejaVuSans.ttf", size=dynamic_size)

                        except IOError:

                            font = ImageFont.load_default()



                    for item in detected_items:

                        w = item.get('width')

                        h = item.get('height')

                        x0 = item.get('x') - (w / 2)

                        y0 = item.get('y') - (h / 2)

                        x1 = item.get('x') + (w / 2)

                        y1 = item.get('y') + (h / 2)

                       

                        label = f"{item.get('class')} ({item.get('confidence'):.0%})"

                       

                        # Draw bounding box outline using our dynamic box_width

                        draw.rectangle([x0, y0, x1, y1], outline="#00FF00", width=box_width)

                       

                        # Automatically calculate text dimensions for the background box

                        try:

                            # Pillow 10.0.0+ method

                            text_w, text_h = draw.textbbox((0, 0), label, font=font)[2:]

                        except AttributeError:

                            # Older Pillow versions fallback

                            text_w, text_h = draw.textsize(label, font=font)

                           

                        # Draw a small text background box and label text relative to text heights

                        draw.rectangle([x0, y0 - text_h - 6, x0 + text_w + 10, y0], fill="#00FF00")

                        draw.text((x0 + 5, y0 - text_h - 4), label, fill="#000000", font=font)

                   

                    # 6. Display the final annotated image

                    st.image(annotated_image, caption="AI Predicted Bounding Boxes", use_container_width=True)

                   

                    # Optional: Detailed breakdown expanders underneath

                    st.subheader("Raw Detection Breakdown")

                    for item in detected_items:

                        with st.expander(f"📦 {item.get('class').title()}"):

                            st.write(f"**Confidence:** {item.get('confidence'):.2%}")

                            st.write(f"**Bounding Box Dimensions:** Width: {item.get('width')}, Height: {item.get('height')}")

                else:

                    st.image(original_image, caption="Uploaded Image", use_container_width=True)

                    st.info("No trash items detected based on current confidence thresholds.")

                   

            except Exception as e:

                st.error(f"An error occurred: {e}")

            finally:

                if os.path.exists(temp_path):

                    os.remove(temp_path)

