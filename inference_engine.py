import cv2
import numpy as np
from openvino.runtime import Core

MODEL_PATH = "best_int8_openvino_model/best.xml"
CLASSES = ["Helmet", "Vest", "Glove", "Person", "Boots"]

try:
    core = Core()
    model_ov = core.read_model(model=MODEL_PATH)
    compiled_model = core.compile_model(model=model_ov, device_name="CPU")
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
except Exception as e:
    print(f"[!] Error loading OpenVINO model: {e}")
    compiled_model = None

def detect_epp(frame):
    """
    Inferencia local con OpenVINO runtime.
    """
    detections = []
    
    if compiled_model is None:
        return detections
        
    try:
        # Preprocesamiento
        h, w = frame.shape[:2]
        input_h, input_w = 640, 640 # Tamaño típico de YOLO
        
        resized = cv2.resize(frame, (input_w, input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = rgb.transpose(2, 0, 1) # HWC to CHW
        input_data = input_data.reshape(1, 3, input_h, input_w).astype(np.float32) / 255.0

        # Inferencia
        results = compiled_model([input_data])[output_layer]
        
        # Postprocesamiento para salida (1, 9, 8400) de YOLO (v8/v11) (4 coords + 5 clases)
        results = np.squeeze(results) # (9, 8400)
        results = results.T # (8400, 9)
        
        boxes = []
        confidences = []
        class_ids = []
        
        for row in results:
            class_scores = row[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            if confidence > 0.4: # Umbral de confianza
                cx, cy, bw, bh = row[:4]
                
                # Escalar coordenadas al tamaño original
                cx = cx * (w / input_w)
                cy = cy * (h / input_h)
                bw = bw * (w / input_w)
                bh = bh * (h / input_h)
                
                x = int(cx - bw / 2)
                y = int(cy - bh / 2)
                
                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))
                
        # Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)
        
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, bw, bh = boxes[i]
                x2 = x + bw
                y2 = y + bh
                detections.append({
                    "class": CLASSES[class_ids[i]],
                    "bbox": (x, y, x2, y2),
                    "conf": confidences[i]
                })
                
    except Exception as e:
        print(f"[!] Error en inferencia OpenVINO: {e}")

    return detections
