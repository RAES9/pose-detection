import cv2
import mediapipe as mp
import json
import time
import os

mp_pose = mp.solutions.pose

class PoseModelCreator:
    def __init__(self):
        self.pose = mp_pose.Pose()
        self.pose_models = []

    def process_frame(self, frame):
        """Procesar un frame y obtener los puntos clave"""
        try:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            points_detected = [None] * 33
            if results.pose_landmarks:
                height, width, _ = frame.shape
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    points_detected[i] = (landmark.x * width, landmark.y * height)
            return points_detected
        except Exception as e:
            print(f"Error al procesar el frame: {e}")
            return [None] * 33

    def create_pose_model(self, name, frames):
        """Crear un modelo de pose a partir de varios frames capturados"""
        try:
            pose_model = []

            for frame in frames:
                points = self.process_frame(frame)
                if len(points) == 33:
                    pose_model.append(self.generate_validations(points))

            # Guardar la pose con el nombre dado
            self.pose_models.append({
                "message": name,
                "pose_model": self.find_most_frequent_validations(pose_model)
            })
        except Exception as e:
            print(f"Error al crear el modelo de pose para {name}: {e}")

    def generate_validations(self, detected_points):
        """Generar las validaciones a partir de los puntos detectados"""
        validations = []
        try:
            if None not in detected_points:
                validations.append(self.validate_point(detected_points[0], detected_points[12]))  # Nariz a hombro derecho
                validations.append(self.validate_point(detected_points[0], detected_points[9]))  # Nariz a muñeca izquierda
                validations.append(self.validate_point(detected_points[0], detected_points[11]))  # Nariz a hombro izquierdo
                validations.append(self.validate_point(detected_points[0], detected_points[8]))  # Nariz a muñeca derecha

                # Hombro derecho a otras partes
                validations.append(self.validate_point(detected_points[12], detected_points[9]))  # Hombro derecho a muñeca izquierda
                validations.append(self.validate_point(detected_points[12], detected_points[15]))  # Hombro derecho a codo derecho
                validations.append(self.validate_point(detected_points[12], detected_points[18]))  # Hombro derecho a rodilla derecha
                validations.append(self.validate_point(detected_points[12], detected_points[16]))  # Hombro derecho a muñeca izquierda

                # Muñeca izquierda a otras partes
                validations.append(self.validate_point(detected_points[9], detected_points[15]))  # Muñeca izquierda a codo izquierdo
                validations.append(self.validate_point(detected_points[9], detected_points[14]))  # Muñeca izquierda a codo derecho
                validations.append(self.validate_point(detected_points[9], detected_points[18]))  # Muñeca izquierda a rodilla izquierda
                validations.append(self.validate_point(detected_points[9], detected_points[17]))  # Muñeca izquierda a rodilla derecha

                # Rodilla derecha y hombros
                validations.append(self.validate_point(detected_points[14], detected_points[17]))  # Rodilla derecha a codo izquierdo
                validations.append(self.validate_point(detected_points[14], detected_points[12]))  # Codo derecho a hombro derecho
                validations.append(self.validate_point(detected_points[14], detected_points[18]))  # Codo derecho a rodilla izquierda
                validations.append(self.validate_point(detected_points[14], detected_points[15]))  # Codo derecho a muñeca izquierda

                # Piernas y pies
                validations.append(self.validate_point(detected_points[17], detected_points[18]))  # Rodilla izquierda a rodilla derecha
                validations.append(self.validate_point(detected_points[17], detected_points[15]))  # Rodilla izquierda a muñeca izquierda
                validations.append(self.validate_point(detected_points[17], detected_points[12]))  # Rodilla izquierda a hombro derecho
                validations.append(self.validate_point(detected_points[17], detected_points[19]))  # Rodilla izquierda a tobillo izquierdo

                # Muñeca derecha a otras partes
                validations.append(self.validate_point(detected_points[19], detected_points[16]))  # Muñeca derecha a codo izquierdo
                validations.append(self.validate_point(detected_points[19], detected_points[9]))  # Muñeca derecha a muñeca izquierda
                validations.append(self.validate_point(detected_points[19], detected_points[12]))  # Muñeca derecha a hombro derecho
                validations.append(self.validate_point(detected_points[19], detected_points[15]))  # Muñeca derecha a muñeca izquierda

                # Validaciones adicionales de piernas
                validations.append(self.validate_point(detected_points[16], detected_points[0]))  # Rodilla izquierda a cabeza
                validations.append(self.validate_point(detected_points[16], detected_points[15]))  # Rodilla izquierda a muñeca derecha
                validations.append(self.validate_point(detected_points[16], detected_points[18]))  # Rodilla izquierda a rodilla derecha
                validations.append(self.validate_point(detected_points[16], detected_points[19]))  # Rodilla izquierda a tobillo derecho

                # Validaciones adicionales de codo y muñeca
                validations.append(self.validate_point(detected_points[8], detected_points[12]))  # Codo izquierdo a hombro derecho
                validations.append(self.validate_point(detected_points[8], detected_points[9]))  # Codo izquierdo a muñeca izquierda
                validations.append(self.validate_point(detected_points[8], detected_points[11]))  # Codo izquierdo a hombro izquierdo
                validations.append(self.validate_point(detected_points[8], detected_points[14]))  # Codo izquierdo a codo derecho

                # Hombro izquierdo con otros puntos
                validations.append(self.validate_point(detected_points[11], detected_points[12]))  # Hombro izquierdo a hombro derecho
                validations.append(self.validate_point(detected_points[11], detected_points[9]))  # Hombro izquierdo a muñeca izquierda
                validations.append(self.validate_point(detected_points[11], detected_points[17]))  # Hombro izquierdo a rodilla izquierda
                validations.append(self.validate_point(detected_points[11], detected_points[14]))  # Hombro izquierdo a codo derecho

                # Validaciones finales entre piernas y pies
                validations.append(self.validate_point(detected_points[13], detected_points[9]))  # Cadera izquierda a muñeca izquierda
                validations.append(self.validate_point(detected_points[13], detected_points[8]))  # Cadera izquierda a codo derecho
                validations.append(self.validate_point(detected_points[13], detected_points[15]))  # Cadera izquierda a rodilla izquierda
                validations.append(self.validate_point(detected_points[13], detected_points[16]))  # Cadera izquierda a rodilla derecha

                validations.append(self.validate_point(detected_points[13], detected_points[12]))  # Cadera izquierda a hombro derecho
                validations.append(self.validate_point(detected_points[13], detected_points[11]))  # Cadera izquierda a hombro izquierdo
                validations.append(self.validate_point(detected_points[13], detected_points[18]))  # Cadera izquierda a rodilla izquierda
                validations.append(self.validate_point(detected_points[13], detected_points[19]))  # Cadera izquierda a tobillo derecho
        except Exception as e:
            print(f"Error generando validaciones: {e}")
            return []

        return validations

    def validate_point(self, point1, point2):
        """Validar un par de puntos clave"""
        if point1 is None or point2 is None:
            return 0

        x_diff = point1[0] - point2[0]
        y_diff = point1[1] - point2[1]
        total = 0

        if x_diff + 0.10 < 0:
            total += 1
        if y_diff + 0.10 < 0:
            total += 2
        if x_diff - 0.10 > 0:
            total += 3
        if y_diff - 0.10 > 0:
            total += 4
        if -0.10 < y_diff < 0.10:
            total += 5
        if -0.10 < x_diff < 0.10:
            total += 6

        return total

    def find_most_frequent_validations(self, validations):
        """Encontrar la validación más frecuente entre los frames"""
        if not validations:
            return []

        frequent_validations = []
        for i in range(len(validations[0])):
            counts = {}
            for validation in validations:
                counts[validation[i]] = counts.get(validation[i], 0) + 1
            frequent_validations.append(max(counts, key=counts.get))

        return frequent_validations

    def save_pose_models(self, filename="pose_models.json"):
        """Guardar los modelos de pose en un archivo JSON"""
        try:
            with open(filename, 'w') as file:
                json.dump(self.pose_models, file)
            print(f"Pose models saved to {filename}")
        except Exception as e:
            print(f"Error al guardar el modelo de poses: {e}")

    def real_time_pose_creation(self):
        """Crear modelos de pose en tiempo real usando la cámara"""
        cap = cv2.VideoCapture(0)
        try:
            while True:
                input("Presiona Enter para comenzar la creación de una nueva pose...")
                name = input("Introduce el nombre de la pose: ")

                print("Ponte en posición, comenzando en 3 segundos...")
                time.sleep(3)

                frames = []
                print("Capturando frames durante 10 segundos...")
                start_time = time.time()
                while time.time() - start_time < 10:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                    cv2.imshow('Capturando Pose', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                self.create_pose_model(name, frames)

                cont = input("¿Deseas agregar otra pose? (s/n): ")
                if cont.lower() != 's':
                    break
        except Exception as e:
            print(f"Error en la creación de pose en tiempo real: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.save_pose_models()

    def create_pose_from_images(self, dataset_path):
        """Crear modelos de pose desde imágenes en un dataset"""
        try:
            for pose_name in os.listdir(dataset_path):
                pose_path = os.path.join(dataset_path, pose_name)
                if os.path.isdir(pose_path):
                    frames = [cv2.imread(os.path.join(pose_path, img)) for img in os.listdir(pose_path)]
                    self.create_pose_model(pose_name, frames)

            self.save_pose_models()
        except Exception as e:
            print(f"Error al crear poses desde imágenes: {e}")