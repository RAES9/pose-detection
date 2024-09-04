import unittest
import cv2
from pose_detection.pose_model_creator import PoseModelCreator
from pose_detection.pose_detector import PoseDetector


class TestPoseDetection(unittest.TestCase):

    def test_pose_model_generation(self):
        """Prueba la generación de modelos de poses desde imágenes"""
        generator = PoseModelCreator()
        dataset_path = 'test_dataset'  # Ruta del dataset de prueba

        # Procesar imágenes desde el dataset
        generator.process_images_in_dataset(dataset_path)

        # Comprobar si los modelos de pose se generaron correctamente
        self.assertGreater(len(generator.pose_model), 0)
        self.assertEqual(len(generator.pose_model[0]), 16)  # Validaciones esperadas

    def test_pose_detection(self):
        """Prueba la detección de poses con el modelo generado"""
        # Crear un modelo de prueba
        generator = PoseModelCreator()
        dataset_path = 'test_dataset'
        generator.process_images_in_dataset(dataset_path)

        # Guardar el modelo en un archivo JSON
        generator.generate_pose_model()
        detector = PoseDetector("pose_models.json")

        # Prueba la detección de una pose en tiempo real
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        self.assertTrue(ret)

        detected_points = detector.process_frame(frame)
        pose_name, confidence = detector.validate_pose(detected_points)
        cap.release()

        # Comprobar si se detectó una pose con cierta confianza
        self.assertTrue(confidence > 0)
        print(f"Pose: {pose_name}, Confidence: {confidence}%")


if __name__ == "__main__":
    unittest.main()