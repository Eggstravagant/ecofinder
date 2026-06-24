import streamlit as st

# 1. Page Configuration
st.set_page_config(page_title="Live TACO Trash Detector", layout="centered")
st.title("🗑️ Live TACO Trash Detector")
st.write("Real-time webcam feed analyzing waste using your browser.")

# 2. Embed Local Browser-Based Live Scanner Component
# This runs the detection loop in JavaScript directly inside the browser window
live_scanner_html = """
<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 15px;">
    <div style="position: relative; width: 100%; max-width: 640px; aspect-ratio: 4/3; background: #000; border-radius: 8px; overflow: hidden;">
        <video id="webcam" autoplay playsinline style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; z-index: 1;"></video>
        <canvas id="overlay" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; z-index: 2;"></canvas>
    </div>
    <div id="status" style="font-family: sans-serif; font-weight: bold; color: #4CAF50;">Initializing Camera...</div>
</div>

<script>
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('overlay');
    const ctx = canvas.getContext('2d');
    const statusText = document.getElementById('status');

    const API_KEY = "zs2Fpdr6GYFftsJ7kQws";
    const WORKFLOW_ID = "trash-detection-workflow-1782289148660";
    const WORKSPACE = "dylans-workspace-gchst";

    // Setup local user webcam
    navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false })
        .then(stream => {
            video.srcObject = stream;
            video.addEventListener('loadedmetadata', () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                statusText.innerText = "⚡ Live Feed Active - Analyzing Frames...";
                // Start processing loops
                requestAnimationFrame(processFrame);
            });
        })
        .catch(err => {
            statusText.innerText = "❌ Camera Error: Access Denied or Not Found";
            statusText.style.color = "#FF3B30";
        });

    let processing = false;

    async function processFrame() {
        if (processing) {
            requestAnimationFrame(processFrame);
            return;
        }
        processing = true;

        // Draw current video frame to hidden state to turn into a blob package
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = canvas.width;
        tempCanvas.height = canvas.height;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(video, 0, 0, canvas.width, canvas.height);

        tempCanvas.toBlob(async (blob) => {
            if (!blob) {
                processing = false;
                requestAnimationFrame(processFrame);
                return;
            }

            // Construct our workflow call format package
            const formData = new FormData();
            formData.append('image', blob, 'frame.jpg');

            try {
                // Send current live frame directly over network
                const response = await fetch(`https://serverless.roboflow.com/${WORKSPACE}/workflows/${WORKFLOW_ID}?api_key=${API_KEY}`, {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                
                // Safely clear out the last frame's bounding boxes
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // Unpack deep target predictions list
                let results = [];
                if (Array.isArray(data)) {
                    results = data[0]?.predictions?.predictions || [];
                } else if (data?.predictions) {
                    results = data.predictions.predictions || [];
                }

                // Render bounding boxes over live elements
                results.forEach(item => {
                    const w = item.width;
                    const h = item.height;
                    const x0 = item.x - (w / 2);
                    const y0 = item.y - (h / 2);

                    // Draw styling box
                    ctx.strokeStyle = '#00FF00';
                    ctx.lineWidth = Math.max(3, canvas.width * 0.005);
                    ctx.strokeRect(x0, y0, w, h);

                    // Draw text background block
                    const label = `${item.class} (${Math.round(item.confidence * 100)}%)`;
                    ctx.fillStyle = '#00FF00';
                    ctx.font = `${Math.max(14, canvas.width * 0.025)}px Arial`;
                    const textWidth = ctx.measureText(label).width;
                    ctx.fillRect(x0, y0 - (canvas.width * 0.035), textWidth + 10, canvas.width * 0.035);

                    // Set text color
                    ctx.fillStyle = '#000000';
                    ctx.fillText(label, x0 + 5, y0 - 6);
                });

            } catch (error) {
                console.error("Inference Error:", error);
            }

            processing = false;
            // Introduce a short timeout to prevent spamming your API credits instantly
            setTimeout(() => { requestAnimationFrame(processFrame); }, 150);
        }, 'image/jpeg', 0.7);
    }
</script>
"""

# 3. Mount component with container stretching widths
st.components.v1.html(live_scanner_html, height=520, scrolling=False)