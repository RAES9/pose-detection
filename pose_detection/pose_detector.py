import cv2
import mediapipe as mp
import json

mp_pose = mp.solutions.pose


class PoseDetector:
    def __init__(self, model_file=None):
        self.pose = mp_pose.Pose()
        self.pose_model = []

        if model_file:
            self.load_pose_model(model_file)

    def load_pose_model(self, model_file):
        """Cargar el modelo de pose desde un archivo JSON"""
        with open(model_file, 'r') as file:
            self.pose_model = json.load(file)

    def process_frame(self, frame):
        """Procesar un frame para obtener las poses"""
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
            print(f"Error procesando el frame: {e}")
            return [None] * 33

    def validate_pose(self, detected_points):
        """Validar la pose detectada con el modelo cargado"""
        if len(detected_points) != 33 or None in detected_points:
            print("Error: Datos incompletos para la detección de pose.")
            return None, 0

        for pose_reference in self.pose_model:
            detected, confidence = self._validate_single_pose(detected_points, pose_reference)
            if detected:
                return pose_reference['message'], confidence

        return None, 0

    def _validate_single_pose(self, detected_points, pose_reference):
        """Comparar puntos detectados con los modelos guardados"""
        validations_detected = self.generate_validations(detected_points)
        pose_model = pose_reference['pose_model']

        if len(validations_detected) != len(pose_model):
            print("Error: El número de validaciones no coincide con el modelo de la pose.")
            return False, 0

        counter = 0
        counter_failed = 0

        try:
            for i in range(len(pose_model)):
                if validations_detected[i] == pose_model[i]:
                    counter += 1
                else:
                    counter_failed += 1
                    if counter_failed == 5:
                        return False, 0
        except IndexError:
            print("Error: Se produjo un IndexError al validar los puntos.")
            return False, 0

        confidence = (counter / len(pose_model)) * 100
        return counter > 10, confidence

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