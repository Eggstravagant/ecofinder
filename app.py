import cv2
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont

# 1. Connect to Roboflow Client
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="zs2Fpdr6GYFftsJ7kQws"
)

# 2. Initialize your local webcam (0 is usually your default built-in camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("🚀 Live Trash Detection Started! Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame from BGR (OpenCV) to RGB (PIL)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    try:
        # 3. Send frame to Roboflow Workflow Framework
        result = client.run_workflow(
            workspace_name="dylans-workspace-gchst",
            workflow_id="trash-detection-workflow-1782289148660",
            images={"image": pil_img},
            parameters={"classes": "paper, plastic, glass, metal, cardboard"},
            use_cache=True
        )

        # Parse data blocks
        workflow_output = result[0] if isinstance(result, list) else result
        predictions_data = workflow_output.get("predictions", {})
        detected_items = predictions_data.get("predictions", [])

        # 4. Draw bounding boxes on the frame
        for item in detected_items:
            w, h = item.get('width'), item.get('height')
            x0 = item.get('x') - (w / 2)
            y0 = item.get('y') - (h / 2)
            x1 = item.get('x') + (w / 2)
            y1 = item.get('y') + (h / 2)

            label = f"{item.get('class')} ({item.get('confidence'):.0%})"
            
            # Draw green rectangle box
            draw.rectangle([x0, y0, x1, y1], outline="#00FF00", width=3)
            draw.text((x0 + 5, y0 - 15), label, fill="#00FF00")

    except Exception as e:
        # If a single frame request times out, skip it gracefully to prevent screen freeze
        pass

    # Convert back to standard OpenCV format to display on desktop window screen
    final_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR) if 'np' in globals() else frame
    
    # Alternatively, parse directly with OpenCV format safely
    cv2.imshow('TACO Live Trash Detection', final_frame)

    # Break the loop when the user presses 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up window instances
cap.release()
cv2.destroyAllWindows()