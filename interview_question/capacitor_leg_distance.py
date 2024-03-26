# Tarık Bacak
# bacak21@itu.edu.tr
# 22.03.2024
#
# This script measures the distance
# between the closest pair of target points
# on the legs of a capacitor.


import cv2
import numpy as np
import time
import typing


class CapacitorLegDistance:
    def __init__(self, image_path: str) -> None:
        """
        This function initializes the class.
        :param image_path: Path to the image
        """
        # Path to the image
        self.image_path = image_path

        # ROI coordinates
        self.roi_start_y = 480
        self.roi_end_y = 680
        self.roi_start_x = 0
        self.roi_end_x = 1600

        # Canny edge detection thresholds
        self.threshold_1 = 200
        self.threshold_2 = 200

        # Counter for modifying the y-level
        self.counter = 0

        # The image to be processed
        self.image = None

    def __load_image(self) -> None:
        """
        This function loads the image.
        """
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image '{self.image_path}' not found.")

    def __convert_to_grayscale(self) -> cv2.typing.MatLike:
        """
        This function converts the image to grayscale.
        :return: Grayscale image
        """
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return gray_image

    def __edge_detection(self, gray_image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        """
        This function applies Canny edge detection to the grayscale image.
        :param gray_image: Grayscale image
        :return: Edges of the image
        """
        edges = cv2.Canny(gray_image, self.threshold_1, self.threshold_2)
        return edges

    def __find_leg_contours(self, edges: cv2.typing.MatLike) -> typing.Sequence[cv2.typing.MatLike]:
        """
        This function finds the contours in the ROI.
        :param edges: Edges of the image
        :return: List of contours
        """
        # Define the ROI
        roi = edges[self.roi_start_y:self.roi_end_y, self.roi_start_x:self.roi_end_x]

        # Draw the ROI on the image and put the text
        cv2.rectangle(self.image, (self.roi_start_x, self.roi_start_y),
                      (self.roi_end_x, self.roi_end_y), (255, 0, 0), 2)
        cv2.putText(self.image, "ROI", (self.roi_start_x, self.roi_start_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Find contours in the ROI
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        offset_contours = [contour + np.array([self.roi_start_x, self.roi_start_y]) for contour in contours]
        cv2.drawContours(self.image, offset_contours, -1, (255, 0, 255), 1)

        return contours

    def __find_target_points(self, contours: typing.Sequence[cv2.typing.MatLike]) -> list[tuple[int, int]]:
        """
        This function finds the target points on the target y-levels.
        :param contours: List of contours
        :return: List of target points
        """
        target_points = []

        for contour in contours:
            # Find the bottom point of the contour
            bottom_point = max(contour, key=lambda x: x[0][1])

            # Findig the target y-level
            target_y = bottom_point[0][1] - 20
            print(f"\ntarget_y:   {target_y}")

            # Find the middle point between the leftmost and rightmost edges of the contour at the target y-level
            self.counter = 0
            middle_x = self.__find_middle_of_edges_at_y(contour, target_y)
            print(f"middle_x:   {middle_x}")

            # Coordinates of the target point
            point = (middle_x + self.roi_start_x, target_y + self.roi_start_y)
            target_points.append(point)

            # Draw the target point
            cv2.circle(self.image, point, 3, (0, 255, 0), 2)

        return target_points

    def __find_middle_of_edges_at_y(self, contour: cv2.typing.MatLike, y: int) -> int:
        """
        This function finds the middle point between the leftmost and rightmost edges of a contour at a given y-level.
        :param contour: Contour
        :param y: Y-level
        :return: x-coordinate of the middle point
        """
        # Extract all points at the given y-level
        points_at_y = [point[0] for point in contour if point[0][1] == y]

        # If no points found at this y-level, return the modified y-level and call the function recursively
        if not points_at_y:
            y = self.__modify_y(y)
            print(f"modified y: {y}")
            return self.__find_middle_of_edges_at_y(contour, y)

        # Find the most left and right points at the given y-level
        leftmost = min(points_at_y, key=lambda point: point[0])
        rightmost = max(points_at_y, key=lambda point: point[0])

        # If the leftmost and rightmost points are the same, return the modified y-level and call function recursively
        if leftmost[0] == rightmost[0]:
            y = self.__modify_y(y)
            print(f"modified y: {y}")
            return self.__find_middle_of_edges_at_y(contour, y)

        # Calculate the middle x
        middle_x = (leftmost[0] + rightmost[0]) // 2

        return middle_x

    def __modify_y(self, y: int) -> int:
        """
        This function modifies the y-level for the next iteration.
        Examples: y+1, y-1, y+2, y-2, ...
        :param y: Current y-level
        :return: Modified y-level
        """
        # Increase or decrease the y-level based on the counter
        self.counter += 1

        # If the counter is odd, increase the y-level, otherwise decrease
        if self.counter % 2 == 1:
            new_y = y + self.counter
        else:
            new_y = y - self.counter

        return new_y

    def __measure_distance(self, target_points: list[tuple[int, int]]) -> None:
        """
        This function measures the distance between the closest pair of target points.
        :param target_points: List of target points
        """
        # If there are at least two target points, find the closest pair and measure the distance
        if len(target_points) > 1:
            distance_between_points = float('inf')
            closest_pair = None

            # Find the closest pair of target points
            for i in range(len(target_points)):
                for j in range(i + 1, len(target_points)):
                    distance = np.linalg.norm(np.array(target_points[i]) - np.array(target_points[j]))
                    if distance < distance_between_points:
                        distance_between_points = distance
                        closest_pair = (target_points[i], target_points[j])

            # Draw the distance line and write the distance value on the image
            if closest_pair:
                distance_text = f"Distance: {distance_between_points:.1f} px"
                cv2.putText(self.image, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.line(self.image, closest_pair[0], closest_pair[1], (0, 0, 255), 2)

    def __display_results(self, duration: float) -> None:
        """
        This function displays the measurement results.
        :param duration: Duration of the process
        """
        # Draw the duration on the image
        duration_text = f"Duration: {duration:.3f} s"
        cv2.putText(self.image, duration_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the measurement results
        cv2.imshow('Measurement', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("measurement_result.jpg", self.image)

    def run(self) -> None:
        """
        This function runs the whole process.
        """
        start_time = time.time()

        self.__load_image()
        gray_image = self.__convert_to_grayscale()
        edges = self.__edge_detection(gray_image)
        contours = self.__find_leg_contours(edges)
        target_points = self.__find_target_points(contours)
        self.__measure_distance(target_points)

        end_time = time.time()
        duration = end_time - start_time
        self.__display_results(duration)


if __name__ == "__main__":
    img_path = "IP1_Cap.jpg"
    capacitor_leg_distance = CapacitorLegDistance(image_path=img_path)
    capacitor_leg_distance.run()
