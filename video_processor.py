import cv2
import threading
import queue
from inference_engine import detect_epp
from alert_system import AlertSystem

class VideoProcessor:
    def __init__(self, source, frame_skip=5):
        self.source = source
        self.frame_skip = frame_skip
        self.cap = cv2.VideoCapture(source)
        
        # Preprocesamiento: Ecualización de histograma adaptativa (CLAHE)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Colas para comunicación asíncrona entre captura e inferencia
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        
        self.running = False
        self.alert_system = AlertSystem()
        
        # Compartir estado de forma segura entre hilos
        self.latest_payload = {"detections": [], "is_violation": False, "missing_epps": []}
        self.detection_lock = threading.Lock()
        
        self.frame_count = 0

    def preprocess_light(self, frame):
        """
        Aplica CLAHE en el canal de luminancia (L de espacio LAB)
        para estandarizar iluminación extrema (obras/interiores).
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    def inference_worker(self):
        """Hilo en background (Worker Thread) para ejecutar el modelo de inferencia."""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                # Inferencia real con YOLO OpenVINO/ONNX
                detections = detect_epp(frame)
                
                # Lógica de Seguridad Proactiva
                detected_labels = set(d["class"] for d in detections)
                missing_epps = []
                is_violation = False
                
                # Filtrado espacial: Solo evaluar faltantes si hay al menos una persona
                if "Person" in detected_labels:
                    if "Helmet" not in detected_labels:
                        missing_epps.append("Helmet")
                    if "Vest" not in detected_labels:
                        missing_epps.append("Vest")
                    if "Glove" not in detected_labels:
                        missing_epps.append("Glove")
                    if "Boots" not in detected_labels:
                        missing_epps.append("Boots")
                
                # Disparar alerta si aplica
                if missing_epps:
                    is_violation = True
                    self.alert_system.trigger_alert(frame, missing_epps, detected_labels)

                payload = {
                    "detections": detections,
                    "is_violation": is_violation,
                    "missing_epps": missing_epps
                }

                # Gestión de Resultados: Limpiar cola llena para garantizar que 
                # la UI reciba la detección más reciente.
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                # Enviar payload actualizado
                self.result_queue.put(payload)
                    
            except queue.Empty:
                continue

    def start(self):
        self.running = True
        threading.Thread(target=self.inference_worker, daemon=True).start()

    def stop(self):
        self.running = False
        self.cap.release()

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, None

        # Obtener las últimas detecciones listas (si existen)
        try:
            new_payload = self.result_queue.get_nowait()
            # Sincronización de Hilos: Lock para evitar race conditions
            with self.detection_lock:
                self.latest_payload = new_payload
        except queue.Empty:
            pass

        self.frame_count += 1
        
        # Frame Skipping: Enviar solo 1 de cada N frames a inferencia
        if self.frame_count % self.frame_skip == 0:
            if not self.frame_queue.full():
                # Aplicar preprocesamiento de iluminación solo para inferencia
                processed_frame = self.preprocess_light(frame)
                self.frame_queue.put(processed_frame)

        # UI Cruda: Dibujar Bounding Boxes en el frame original para máxima fluidez
        display_frame = frame.copy()
        
        # Leer estado de forma segura
        with self.detection_lock:
            current_payload = dict(self.latest_payload)
            
        current_detections = current_payload.get("detections", [])
        is_violation = current_payload.get("is_violation", False)
        missing_epps = current_payload.get("missing_epps", [])

        # UI Visual: Alerta roja si hay infracción
        if is_violation:
            h, w = display_frame.shape[:2]
            cv2.rectangle(display_frame, (0, 0), (w, h), (0, 0, 255), 6)
            cv2.putText(display_frame, f"ALERTA: Faltan {', '.join(missing_epps)}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
        for det in current_detections:
            x1, y1, x2, y2 = det["bbox"]
            cls = det["class"]
            conf = det["conf"]
            
            # Color rojo si una persona está en infracción
            if cls == "Person" and is_violation:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0) # Verde normal
                
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, f"{cls} {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        return True, display_frame