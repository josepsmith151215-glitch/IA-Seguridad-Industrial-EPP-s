import cv2
import argparse
from video_processor import VideoProcessor

def main():
    parser = argparse.ArgumentParser(description="Detección de EPP en tiempo real (Optimizado CPU)")
    parser.add_argument("--source", type=str, default="0", 
                        help="Fuente (Ej: 0 para webcam local, rtsp://ip/stream para cámaras IP)")
    parser.add_argument("--skip", type=int, default=5, 
                        help="Frame skip para inferencia (Ej: 5 = procesar 1 de cada 5 frames)")
    args = parser.parse_args()

    # Parsear fuente: usar entero si es un dígito (webcam local)
    source = int(args.source) if args.source.isdigit() else args.source

    print(f"[*] Iniciando detección EPP. Fuente: {source} | Frame Skip: {args.skip}")
    
    processor = VideoProcessor(source=source, frame_skip=args.skip)
    processor.start()

    window_name = "EPP Detection (CPU Optimized)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("[*] Presione 'q' en la ventana de video para salir.")
    try:
        while True:
            ret, frame = processor.get_frame()
            if not ret:
                print("[!] Fallo al capturar video o fin de la transmisión.")
                break

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\n[*] Interrupción manual del usuario.")
    finally:
        processor.stop()
        cv2.destroyAllWindows()
        print("[*] Sistema finalizado.")

if __name__ == "__main__":
    main()
