import os, cv2

video_path = "data/raw_videos/prueba1.mp4"
out_dir = "data/extracted_frames/prueba1"
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
step = max(1, int(fps // 2))  # ~2 frames/seg (rápido y suficiente Día 1)

i = 0
saved = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if i % step == 0:
        t = i / fps
        name = f"frame_{i:06d}_t{t:.2f}.jpg"
        cv2.imwrite(os.path.join(out_dir, name), frame)
        saved += 1
    i += 1

cap.release()
print(f"✅ FPS={fps:.2f} | Frames guardados={saved}")
