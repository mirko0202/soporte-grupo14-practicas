"""Manual door selection and person crossing counter for videos.

Workflow:
1. Open the video.
2. Show the first frame and let the user draw fixed door rectangles with the mouse.
3. Press ENTER or SPACE to start processing.
4. Detect people with YOLO, track them across frames, and count crossings per door.
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - checked at runtime
    YOLO = None


Box = Tuple[int, int, int, int]
Point = Tuple[int, int]


@dataclass
class AppConfig:
	person_confidence: float = 0.35
	tracker_iou_threshold: float = 0.18
	tracker_distance_threshold: float = 0.55
	max_missed_frames: int = 12
	min_person_age_for_count: int = 3
	min_person_hits_for_count: int = 2
	counting_zone_ratio: float = 0.10
	counting_rearm_ratio: float = 0.20
	min_door_width: int = 20
	min_door_height: int = 20
	selection_window: str = "Select doors"
	playback_window: str = "Door Counter"
	selection_instructions: str = (
		"Drag with the mouse to draw each door. ENTER/SPACE to start. "
		"U undo, C clear, ESC cancel."
	)
	people_color: Tuple[int, int, int] = (0, 165, 255)
	doors_color: Tuple[int, int, int] = (70, 200, 70)
	zone_color: Tuple[int, int, int] = (255, 210, 80)
	label_color: Tuple[int, int, int] = (245, 245, 245)
	panel_color: Tuple[int, int, int] = (18, 18, 18)
	font_scale: float = 0.6
	line_thickness: int = 2


@dataclass
class Detection:
	bbox: Box
	score: float


@dataclass
class DoorRegion:
	door_id: int
	bbox: Box
	count: int = 0

	def counting_zone(self, config: AppConfig) -> Box:
		x1, y1, x2, y2 = self.bbox
		inset_x = max(4, int(box_width(self.bbox) * config.counting_zone_ratio))
		inset_y = max(4, int(box_height(self.bbox) * config.counting_zone_ratio))
		zone_x1 = min(x2 - 1, x1 + inset_x)
		zone_y1 = min(y2 - 1, y1 + inset_y)
		zone_x2 = max(zone_x1 + 1, x2 - inset_x)
		zone_y2 = max(zone_y1 + 1, y2 - inset_y)
		return zone_x1, zone_y1, zone_x2, zone_y2


@dataclass
class PersonTrack:
	track_id: int
	bbox: Box
	score: float
	age: int = 1
	hits: int = 1
	missed_frames: int = 0
	active: bool = True
	history: Deque[Point] = field(default_factory=lambda: deque(maxlen=8))

	def bottom_center(self) -> Point:
		x1, y1, x2, y2 = self.bbox
		return int((x1 + x2) / 2), y2


@dataclass
class CrossingState:
	last_side: int = 0
	armed: bool = True


def clamp(value: float, lower: float, upper: float) -> float:
	return max(lower, min(upper, value))


def clip_box(box: Box, frame_width: int, frame_height: int) -> Box:
	x1, y1, x2, y2 = box
	x1 = int(clamp(x1, 0, frame_width - 1))
	y1 = int(clamp(y1, 0, frame_height - 1))
	x2 = int(clamp(x2, 0, frame_width - 1))
	y2 = int(clamp(y2, 0, frame_height - 1))
	if x2 <= x1:
		x2 = min(frame_width - 1, x1 + 1)
	if y2 <= y1:
		y2 = min(frame_height - 1, y1 + 1)
	return x1, y1, x2, y2


def box_width(box: Box) -> int:
	return max(1, box[2] - box[0])


def box_height(box: Box) -> int:
	return max(1, box[3] - box[1])


def box_area(box: Box) -> int:
	return box_width(box) * box_height(box)


def box_center(box: Box) -> Point:
	x1, y1, x2, y2 = box
	return int((x1 + x2) / 2), int((y1 + y2) / 2)


def point_in_box(point: Point, box: Box) -> bool:
	x, y = point
	x1, y1, x2, y2 = box
	return x1 <= x <= x2 and y1 <= y <= y2


def box_bottom_center(box: Box) -> Point:
	x1, y1, x2, y2 = box
	return int((x1 + x2) / 2), y2


def bbox_iou(box_a: Box, box_b: Box) -> float:
	ax1, ay1, ax2, ay2 = box_a
	bx1, by1, bx2, by2 = box_b
	inter_x1 = max(ax1, bx1)
	inter_y1 = max(ay1, by1)
	inter_x2 = min(ax2, bx2)
	inter_y2 = min(ay2, by2)
	if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
		return 0.0
	inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
	union_area = box_area(box_a) + box_area(box_b) - inter_area
	return inter_area / union_area if union_area > 0 else 0.0


def line_side(value: int, line_value: int, dead_zone: int) -> int:
	if value < line_value - dead_zone:
		return -1
	if value > line_value + dead_zone:
		return 1
	return 0


def segment_intersects_box(start: Point, end: Point, box: Box) -> bool:
	"""Liang-Barsky clipping test: true when the segment touches the box."""

	x1, y1, x2, y2 = box
	x0, y0 = start
	x1s, y1s = end
	dx = x1s - x0
	dy = y1s - y0

	p = (-dx, dx, -dy, dy)
	q = (x0 - x1, x2 - x0, y0 - y1, y2 - y0)
	u1 = 0.0
	u2 = 1.0

	for pi, qi in zip(p, q):
		if pi == 0:
			if qi < 0:
				return False
			continue
		t = qi / pi
		if pi < 0:
			if t > u2:
				return False
			if t > u1:
				u1 = t
		else:
			if t < u1:
				return False
			if t < u2:
				u2 = t
	return u1 <= u2


def nms(detections: Sequence[Detection], threshold: float = 0.45) -> List[Detection]:
	if not detections:
		return []
	ordered = sorted(detections, key=lambda item: item.score, reverse=True)
	kept: List[Detection] = []
	while ordered:
		current = ordered.pop(0)
		kept.append(current)
		ordered = [item for item in ordered if bbox_iou(current.bbox, item.bbox) < threshold]
	return kept


def greedy_match(
	tracks: Sequence[PersonTrack],
	detections: Sequence[Detection],
	*,
	iou_threshold: float,
	distance_threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
	active_indices = [index for index, track in enumerate(tracks) if track.active]
	candidates: List[Tuple[float, int, int]] = []

	for track_index in active_indices:
		track = tracks[track_index]
		track_point = track.bottom_center()
		track_diag = max(1.0, math.hypot(box_width(track.bbox), box_height(track.bbox)))
		for detection_index, detection in enumerate(detections):
			detection_point = box_bottom_center(detection.bbox)
			distance = math.dist(track_point, detection_point) / track_diag
			iou = bbox_iou(track.bbox, detection.bbox)
			if iou < iou_threshold and distance > distance_threshold:
				continue
			candidates.append((distance - (0.25 * iou), track_index, detection_index))

	candidates.sort(key=lambda item: item[0])
	matched_tracks: set[int] = set()
	matched_detections: set[int] = set()
	matches: List[Tuple[int, int]] = []

	for _, track_index, detection_index in candidates:
		if track_index in matched_tracks or detection_index in matched_detections:
			continue
		matched_tracks.add(track_index)
		matched_detections.add(detection_index)
		matches.append((track_index, detection_index))

	unmatched_tracks = [index for index in active_indices if index not in matched_tracks]
	unmatched_detections = [index for index in range(len(detections)) if index not in matched_detections]
	return matches, unmatched_tracks, unmatched_detections


class PersonDetector:
	def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.35) -> None:
		if YOLO is None:
			raise RuntimeError("ultralytics is required. Install it with: pip install ultralytics")
		self.model = YOLO(model_path)
		self.confidence_threshold = confidence_threshold

	def detect(self, frame: np.ndarray) -> List[Detection]:
		results = self.model.predict(frame, verbose=False, conf=self.confidence_threshold, classes=[0])
		detections: List[Detection] = []
		for result in results:
			for box in result.boxes:
				x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
				detections.append(Detection((x1, y1, x2, y2), float(box.conf.item())))
		return nms(detections)


class PersonTracker:
	def __init__(self, config: AppConfig) -> None:
		self.config = config
		self.tracks: Dict[int, PersonTrack] = {}
		self.next_id = 1

	def update(self, detections: Sequence[Detection], frame_shape: Tuple[int, int, int]) -> List[PersonTrack]:
		height, width = frame_shape[:2]
		active_tracks = [track for track in self.tracks.values() if track.active]
		matches, unmatched_tracks, unmatched_detections = greedy_match(
			active_tracks,
			detections,
			iou_threshold=self.config.tracker_iou_threshold,
			distance_threshold=self.config.tracker_distance_threshold,
		)

		for active_index, detection_index in matches:
			track = active_tracks[active_index]
			detection = detections[detection_index]
			track.bbox = clip_box(detection.bbox, width, height)
			track.score = detection.score
			track.age += 1
			track.hits += 1
			track.missed_frames = 0
			track.history.append(box_bottom_center(track.bbox))

		for active_index in unmatched_tracks:
			track = active_tracks[active_index]
			track.missed_frames += 1
			if track.missed_frames > self.config.max_missed_frames:
				track.active = False

		for detection_index in unmatched_detections:
			detection = detections[detection_index]
			track = PersonTrack(
				track_id=self.next_id,
				bbox=clip_box(detection.bbox, width, height),
				score=detection.score,
			)
			track.history.append(box_bottom_center(track.bbox))
			self.tracks[self.next_id] = track
			self.next_id += 1

		return [track for track in sorted(self.tracks.values(), key=lambda item: item.track_id) if track.active]


class CrossingCounter:
	def __init__(self, config: AppConfig) -> None:
		self.config = config
		self.states: Dict[Tuple[int, int], CrossingState] = {}

	def update(self, doors: Sequence[DoorRegion], people: Sequence[PersonTrack]) -> List[int]:
		changed_doors: set[int] = set()
		for person in people:
			if not person.history:
				continue
			if person.age < self.config.min_person_age_for_count or person.hits < self.config.min_person_hits_for_count:
				continue
			current_point = person.history[-1]
			previous_point = person.history[-2] if len(person.history) >= 2 else None
			for door in doors:
				state = self.states.setdefault((door.door_id, person.track_id), CrossingState())
				zone = door.counting_zone(self.config)
				current_inside = point_in_box(current_point, zone)
				previous_inside = point_in_box(previous_point, zone) if previous_point is not None else False

				if (
					state.armed
					and previous_point is not None
					and segment_intersects_box(previous_point, current_point, zone)
					and not (previous_inside and current_inside)
				):
					door.count += 1
					changed_doors.add(door.door_id)
					state.armed = False

				zone_center_x, zone_center_y = box_center(zone)
				rearm_distance = max(12, int(max(box_width(zone), box_height(zone)) * self.config.counting_rearm_ratio))
				distance_from_zone = math.dist(current_point, (zone_center_x, zone_center_y))
				if not current_inside and distance_from_zone > rearm_distance:
					state.armed = True
		return sorted(changed_doors)


class DoorSelectionTool:
	def __init__(self, frame: np.ndarray, config: AppConfig) -> None:
		self.base_frame = frame.copy()
		self.config = config
		self.doors: List[DoorRegion] = []
		self.drawing = False
		self.start_point: Optional[Point] = None
		self.current_point: Optional[Point] = None

	def run(self) -> List[DoorRegion]:
		window = self.config.selection_window
		cv2.namedWindow(window, cv2.WINDOW_NORMAL)
		cv2.setMouseCallback(window, self._handle_mouse)

		while True:
			cv2.imshow(window, self._render())
			key = cv2.waitKey(20) & 0xFF
			if key in (13, 32):
				break
			if key == 27:
				self.doors.clear()
				break
			if key in (ord("u"), ord("U")) and self.doors:
				self.doors.pop()
			if key in (ord("c"), ord("C")):
				self.doors.clear()

		cv2.destroyWindow(window)
		return self.doors

	def _handle_mouse(self, event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
		if event == cv2.EVENT_LBUTTONDOWN:
			self.drawing = True
			self.start_point = (x, y)
			self.current_point = (x, y)
		elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
			self.current_point = (x, y)
		elif event == cv2.EVENT_LBUTTONUP and self.drawing:
			self.drawing = False
			self.current_point = (x, y)
			self._commit_rectangle()

	def _commit_rectangle(self) -> None:
		if self.start_point is None or self.current_point is None:
			return
		x1 = min(self.start_point[0], self.current_point[0])
		y1 = min(self.start_point[1], self.current_point[1])
		x2 = max(self.start_point[0], self.current_point[0])
		y2 = max(self.start_point[1], self.current_point[1])
		if (x2 - x1) < self.config.min_door_width or (y2 - y1) < self.config.min_door_height:
			return
		self.doors.append(DoorRegion(door_id=len(self.doors) + 1, bbox=(x1, y1, x2, y2)))

	def _render(self) -> np.ndarray:
		frame = self.base_frame.copy()
		for door in self.doors:
			_draw_door_overlay(frame, door, self.config)
		if self.drawing and self.start_point is not None and self.current_point is not None:
			x1 = min(self.start_point[0], self.current_point[0])
			y1 = min(self.start_point[1], self.current_point[1])
			x2 = max(self.start_point[0], self.current_point[0])
			y2 = max(self.start_point[1], self.current_point[1])
			cv2.rectangle(frame, (x1, y1), (x2, y2), self.config.zone_color, 2)
		_draw_selection_panel(frame, len(self.doors), self.config)
		return frame


def _draw_selection_panel(frame: np.ndarray, door_count: int, config: AppConfig) -> None:
	lines = [config.selection_instructions, f"Doors selected: {door_count}"]
	font = cv2.FONT_HERSHEY_SIMPLEX
	font_scale = 0.55
	thickness = 1
	padding = 10
	line_height = 22
	sizes = [cv2.getTextSize(text, font, font_scale, thickness)[0] for text in lines]
	box_width = max(size[0] for size in sizes) + padding * 2
	box_height = line_height * len(lines) + padding * 2
	overlay = frame.copy()
	cv2.rectangle(overlay, (10, 10), (10 + box_width, 10 + box_height), config.panel_color, -1)
	cv2.addWeighted(overlay, 0.65, frame, 0.35, 0.0, frame)
	for index, text in enumerate(lines):
		y = 10 + padding + (index + 1) * line_height - 6
		cv2.putText(frame, text, (20, y), font, font_scale, config.label_color, thickness, cv2.LINE_AA)


def _draw_door_overlay(frame: np.ndarray, door: DoorRegion, config: AppConfig) -> None:
	x1, y1, x2, y2 = door.bbox
	zone_x1, zone_y1, zone_x2, zone_y2 = door.counting_zone(config)
	overlay = frame.copy()
	cv2.rectangle(overlay, (zone_x1, zone_y1), (zone_x2, zone_y2), config.zone_color, -1)
	cv2.addWeighted(overlay, 0.18, frame, 0.82, 0.0, frame)
	cv2.rectangle(frame, (x1, y1), (x2, y2), config.doors_color, config.line_thickness)
	cv2.putText(frame, f"Door {door.door_id}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, config.font_scale, config.doors_color, 2, cv2.LINE_AA)


def _draw_person_overlay(frame: np.ndarray, person: PersonTrack, config: AppConfig) -> None:
	x1, y1, x2, y2 = person.bbox
	cv2.rectangle(frame, (x1, y1), (x2, y2), config.people_color, 2)
	cv2.putText(frame, f"P{person.track_id}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.people_color, 2, cv2.LINE_AA)


def _draw_counter_panel(frame: np.ndarray, doors: Sequence[DoorRegion], config: AppConfig) -> None:
	lines = [f"Door {door.door_id}: {door.count}" for door in sorted(doors, key=lambda item: item.door_id)]
	if not lines:
		lines = ["No doors selected"]
	font = cv2.FONT_HERSHEY_SIMPLEX
	font_scale = 0.65
	thickness = 1
	padding = 10
	line_height = 24
	sizes = [cv2.getTextSize(text, font, font_scale, thickness)[0] for text in lines]
	box_width = max(size[0] for size in sizes) + padding * 2
	box_height = line_height * len(lines) + padding * 2
	overlay = frame.copy()
	cv2.rectangle(overlay, (10, 10), (10 + box_width, 10 + box_height), config.panel_color, -1)
	cv2.addWeighted(overlay, 0.65, frame, 0.35, 0.0, frame)
	for index, text in enumerate(lines):
		y = 10 + padding + (index + 1) * line_height - 7
		cv2.putText(frame, text, (20, y), font, font_scale, config.label_color, thickness, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, doors: Sequence[DoorRegion], people: Sequence[PersonTrack], config: AppConfig) -> np.ndarray:
	output = frame.copy()
	for door in doors:
		_draw_door_overlay(output, door, config)
	for person in people:
		_draw_person_overlay(output, person, config)
	_draw_counter_panel(output, doors, config)
	return output


class DoorCounterApp:
	def __init__(self, video_path: str, person_model_path: str = "yolov8n.pt", config: Optional[AppConfig] = None) -> None:
		self.video_path = video_path
		self.person_model_path = person_model_path
		self.config = config or AppConfig()
		self.person_detector = PersonDetector(model_path=self.person_model_path, confidence_threshold=self.config.person_confidence)
		self.person_tracker = PersonTracker(self.config)
		self.crossing_counter = CrossingCounter(self.config)
		self.doors: List[DoorRegion] = []

	def run(self) -> int:
		capture = cv2.VideoCapture(self.video_path)
		if not capture.isOpened():
			print(f"Error: could not open video file: {self.video_path}")
			return 1

		success, first_frame = capture.read()
		if not success:
			print(f"Error: could not read the first frame from: {self.video_path}")
			capture.release()
			return 1

		self.doors = DoorSelectionTool(first_frame, self.config).run()
		if not self.doors:
			print("Error: no doors were selected.")
			capture.release()
			cv2.destroyAllWindows()
			return 1

		capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
		cv2.namedWindow(self.config.playback_window, cv2.WINDOW_NORMAL)

		try:
			while True:
				success, frame = capture.read()
				if not success:
					break

				people = self.person_tracker.update(self.person_detector.detect(frame), frame.shape)
				self.crossing_counter.update(self.doors, people)

				cv2.imshow(self.config.playback_window, draw_overlay(frame, self.doors, people, self.config))
				key = cv2.waitKey(1) & 0xFF
				if key in (27, ord("q")):
					break
		except KeyboardInterrupt:
			pass
		finally:
			capture.release()
			cv2.destroyAllWindows()
			self.print_counts()

		return 0

	def print_counts(self) -> None:
		for door in sorted(self.doors, key=lambda item: item.door_id):
			print(f"Door {door.door_id}: {door.count}")


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Manually select doors, track people, and count crossings in a video.")
	parser.add_argument("--video", required=True, help="Path to the input video file.")
	parser.add_argument(
		"--person-model",
		default="yolov8n.pt",
		help="YOLO model path or name for person detection. Default: yolov8n.pt",
	)
	return parser


def main() -> int:
	parser = build_parser()
	args = parser.parse_args()

	video_path = Path(args.video)
	if not video_path.exists():
		print(f"Error: video file not found: {video_path}")
		return 1

	app = DoorCounterApp(video_path=str(video_path), person_model_path=args.person_model)
	return app.run()


if __name__ == "__main__":
	raise SystemExit(main())