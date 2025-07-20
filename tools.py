import cv2
import base64
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

def capture_image():
    """
    Capture one frame from the default webcam, resizes it,
    encodes it as Base64 JPEG(raw string) and returns it.
    """
    for idx in range(4):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            for _ in range(10):
                cap.read()
            ret, frame = cap.read()
            cap.release()
            if not ret:
                continue
            cv2.imwrite("frame.jpg", frame) #Optional
            ret, buf = cv2.imencode(".jpg", frame)
            if ret:
                return base64.b64encode(buf.tobytes()).decode("utf-8")
    raise Exception("No camera found")

capture_image()

def analyze_image_with_query(query: str) -> str:
    """
    Expects a string with query.
    Captures the image and sends the query and the image to
    Groq's vision chat API and returns the analysis.
    """
    img_64 = capture_image()
    model = "meta-llama/llama-4-scout-17b-16e-instruct"

    if not query or not img_64:
        return "Error: both 'query' and 'img_64' must be provided"

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    completion = client.chat.completions.create(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    messages=[
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": query
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{img_64}"
            }
          }
        ]
      },
      
          ],
          
      )

    return completion.choices[0].message.content or ""










