import threading
import os
import time
from datetime import datetime
import cv2

try:
    import winsound
except ImportError:
    winsound = None

class AlertSystem:
    def __init__(self, output_dir="alerts"):
        self.output_dir = output_dir
        self.last_alert_time = 0
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def trigger_alert(self, frame, missing_epps, detected_classes):
        """Dispara la alerta de forma asíncrona validando presencia y cooldown."""
        current_time = time.time()
        
        # a) Se detecta la presencia de un humano
        if "Person" not in detected_classes:
            return
            
        # b) Ha pasado un cooldown de al menos 30 segundos
        if current_time - self.last_alert_time < 30:
            return
            
        self.last_alert_time = current_time

        threading.Thread(
            target=self._process_alert, 
            args=(frame.copy(), missing_epps), 
            daemon=True
        ).start()

    def _process_alert(self, frame, missing_epps):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # 1. Guardar captura de pantalla
        filename = os.path.join(self.output_dir, f"infractor_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        
        # 2. Emitir pitido/alerta sonora del sistema (Windows o fallback silencioso)
        if winsound:
            try:
                winsound.Beep(1000, 500)  # Frecuencia 1000Hz, duración 500ms
            except Exception as e:
                print(f"[!] Error al emitir sonido: {e}")
        else:
            # Fallback para Google Colab o Linux (sonido de sistema estándar o silenciado)
            print("\a", end="")
        
        # 3. Llamar función de SMS
        self._send_mock_sms(filename, missing_epps)

    def _send_mock_sms(self, filepath, missing_epps):
        """
        Integración con API de SMS (ej. Twilio) usando variables de entorno seguras.
        """
        # import twilio.rest
        
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        twilio_phone = os.getenv("TWILIO_PHONE_NUMBER")
        target_phone = os.getenv("TARGET_PHONE_NUMBER")
        
        # if account_sid and auth_token:
        #     client = twilio.rest.Client(account_sid, auth_token)
        #     client.messages.create(
        #         body=f"ALERTA SEGURIDAD: Faltan equipos: {', '.join(missing_epps)}",
        #         from_=twilio_phone,
        #         to=target_phone
        #     )
        #     print(f"[ALERTA ASÍNCRONA] SMS Enviado a {target_phone}: Faltan {missing_epps}.")
        # else:
        print(f"\n[ALERTA ASÍNCRONA] Simulación SMS (Faltan credenciales): Faltan {missing_epps}. Captura: {filepath}\n")
