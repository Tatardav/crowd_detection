import cv2
import torch
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from deep_sort_realtime.deepsort_tracker import DeepSort


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolov8_model_path = 'models/yolov8x.pt'

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=yolov8_model_path,
        confidence_threshold=0.2,
        device=device
    )

    input_filename = 'crowd.mp4'
    output_filename = 'crowd_detected.mp4'

    deepsort = DeepSort(max_age=8, n_init=5, nms_max_overlap=1.0)
    cap = cv2.VideoCapture(input_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = get_sliced_prediction(
            image=frame,
            detection_model=detection_model,
            slice_height=600,
            slice_width=600,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )

        detections = []
        for detection in results.object_prediction_list:
            if detection.category.id == 0:
                x1, y1, x2, y2 = detection.bbox.to_xyxy()
                width, height = x2 - x1, y2 - y1
                confidence = detection.score.value * 100
                det_class = "person"

                if confidence >= 20:
                    detections.append(([x1, y1, width, height], confidence))

        if detections:
            tracks = deepsort.update_tracks(detections, frame=frame)

            for bbox, confidence in detections:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'{det_class} {confidence:.0f}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2)

        if out is None:
            out = cv2.VideoWriter(output_filename, fourcc, cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))
        out.write(frame)

    cap.release()
    out.release()


if __name__ == "__main__":
    main()
