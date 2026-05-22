import os
import sys
import subprocess
import warnings
from collections import defaultdict, deque
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)


def _ensure_dependencies():
    project_root = Path(os.path.dirname(os.path.abspath(__file__)))
    req_path = project_root.parent / "requirements.txt"

    try:
        import tabulate  # noqa: F401
    except ImportError:
        print("[INFO] Installing 'tabulate' for Pandas markdown generation...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tabulate"])

    if req_path.exists():
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(req_path)])
        except subprocess.CalledProcessError:
            sys.exit(1)


_ensure_dependencies()

import cv2
import numpy as np
import supervision as sv
import pandas as pd
from ultralytics import YOLO

# ---------------------------------------------------------
# PERFORMANCE / TRACKING CONFIGURATION
# ---------------------------------------------------------
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
CONFIDENCE_THRESHOLD = 0.25
YOLO_INTERNAL_RESOLUTION = 640
FRAME_SKIP = 1  # state updates every frame; raise only for display if needed

LANE_WIDTH = 3.5
METER_SEGMENT_DISTANCE = 1.0
HEADWAY_REFERENCE_METER = 3

LANE_DEBOUNCE_FRAMES = 5
# Per-axis speed (px/s): both |vx| and |vy| must be below this to count as stationary
WAIT_STATIONARY_PX_PER_S = 25.0

CROP_X_MIN = 335
CROP_Y_MIN = 169
CROP_X_MAX = FRAME_WIDTH
CROP_Y_MAX = FRAME_HEIGHT

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = PROJECT_ROOT.parent.parent / "00_weights"
MODEL_PATH = str(WEIGHTS_DIR / "best.pt")

PED_VEH_DETECTION_DIR = PROJECT_ROOT.parent.parent.parent
VIDEO_PATH = PED_VEH_DETECTION_DIR / "02_clips" / "01_site1" / "01_individual" / "1.mp4"

K = {
    1: (122, 285), 2: (518, 260), 3: (998, 255), 4: (1212, 268),
    5: (1257, 498), 6: (1039, 518), 7: (101, 537),
}

ABOVE_WAITING_POLY = np.array([(1003, 253), (1211, 264), (1199, 206), (993, 189), (1004, 249)], dtype=np.int32)
BELOW_WAITING_POLY = np.array([(1035, 521), (1274, 492), (1274, 560), (1048, 602), (1036, 525)], dtype=np.int32)
DETECTION_POLY = np.array([K[i] for i in [1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
K_AOI_FILTER_POLY = np.array([(54, 579), (55, 232), (1023, 177), (1219, 202), (1273, 509), (1233, 567), (1048, 602)], dtype=np.int32)

TOP_LEFT, TOP_RIGHT = K[1], K[4]
BOTTOM_LEFT, BOTTOM_RIGHT = K[7], K[5]
TOP_Y, BOTTOM_Y = TOP_LEFT[1], BOTTOM_LEFT[1]


def interp(a, b, alpha):
    return (int(a[0] + alpha * (b[0] - a[0])), int(a[1] + alpha * (b[1] - a[1])))


def _is_in_poly(poly, cx, cy):
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) > 0


def _get_short_status_code(overall_status, anchor=None):
    if overall_status == "Crossed":
        return "X"
    if overall_status.startswith("Waiting") and anchor is not None:
        cx, cy = anchor
        if _is_in_poly(ABOVE_WAITING_POLY, cx, cy):
            return "WU"
        if _is_in_poly(BELOW_WAITING_POLY, cx, cy):
            return "WB"
        return "Waiting"
    if "Crossing" in overall_status:
        return overall_status.split("(")[-1].rstrip(")")
    return "N/A"


def sort_ped_statuses(status_list, direction="Down"):
    order_map = {"WB": 1, "R": 2, "M": 3, "L": 4, "X": 5} if direction == "Up" else {"WU": 1, "L": 2, "M": 3, "R": 4, "X": 5}
    return sorted(list(set(status_list)), key=lambda s: order_map.get(s, 99))


class PedestrianAnalyzer:
    def __init__(self, video_path):
        self.video_path = str(video_path)
        self.video_name = Path(video_path).stem

        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"Missing weights file at: {MODEL_PATH}")

        self.model = YOLO(MODEL_PATH)
        self.device = "mps" if subprocess.run(["sysctl", "hw.optional.arm64"], capture_output=True).returncode == 0 else "cpu"
        self.tracker = None

        self.detection_polygon = DETECTION_POLY
        self.lane_lines = self._compute_lane_lines()
        self.lane_coefficients = self._precompute_lane_equations()
        self.grid = self._compute_grid(num_meters=45)
        self.aoi_polygon_coords = K_AOI_FILTER_POLY
        self.aoi_zone = sv.PolygonZone(polygon=self.aoi_polygon_coords, triggering_anchors=[sv.Position.CENTER])

        self.ped_history = {}
        self.ped_overall_status = {}
        self.ped_live_wait_time = {}
        self.is_crossing_started = {}
        self.ped_lane_streak = defaultdict(lambda: deque(maxlen=LANE_DEBOUNCE_FRAMES))

        self.veh_speed_history = {}
        self.veh_segment_cross_time = {}
        self.veh_last_m_band_crossed = {}
        self.veh_first_seen = {}
        self.last_meter_band = {}
        self.last_meter_time = {}
        self.veh_current_lane = {}
        self.veh_time_at_ref_line = {}
        self.live_veh_gaps = {}
        self.last_global_ref_time = None

        self.pair_max_tta = {}
        self.pair_crossing_time = {}
        self.pair_min_dist = {}
        self.pair_status_history = {}
        self.finalized_interactions = []
        self.fps = 30.0

    def _init_tracker(self):
        # supervision >= 0.28 renamed ByteTrack kwargs (track_thresh -> track_activation_threshold, etc.)
        fps = float(self.fps) if self.fps > 0 else 30.0
        lost_buffer = max(30, int(round(fps * 3)))
        try:
            self.tracker = sv.ByteTrack(
                track_activation_threshold=0.45,
                minimum_matching_threshold=0.85,
                lost_track_buffer=lost_buffer,
                frame_rate=fps,
            )
        except TypeError:
            # supervision < 0.28
            self.tracker = sv.ByteTrack(
                track_thresh=0.45,
                match_thresh=0.85,
                track_buffer=lost_buffer,
                frame_rate=int(round(fps)),
            )

    @staticmethod
    def _bbox_center(xyxy):
        return int((xyxy[0] + xyxy[2]) // 2), int((xyxy[1] + xyxy[3]) // 2)

    @staticmethod
    def _ped_anchor(xyxy):
        """Pedestrian centroid at bbox center (not feet)."""
        return PedestrianAnalyzer._bbox_center(xyxy)

    @staticmethod
    def _veh_front_anchor(xyxy):
        x1, y1, x2, y2 = xyxy
        return int(x2), int((y1 + y2) / 2)

    def _compute_lane_lines(self):
        tl, bl = np.array(TOP_LEFT), np.array(BOTTOM_LEFT)
        tr, br = np.array(TOP_RIGHT), np.array(BOTTOM_RIGHT)
        return {
            "L1": (tuple((tl + 0.3333 * (bl - tl)).astype(int)), tuple((tr + 0.3333 * (br - tr)).astype(int))),
            "L2": (tuple((tl + 0.6666 * (bl - tl)).astype(int)), tuple((tr + 0.6666 * (br - tr)).astype(int))),
        }

    def _precompute_lane_equations(self):
        coefs = {}
        for lane_id in ["L1", "L2"]:
            pt_a, pt_b = self.lane_lines[lane_id]
            if pt_b[0] - pt_a[0] != 0:
                m = (pt_b[1] - pt_a[1]) / (pt_b[0] - pt_a[0])
                coefs[lane_id] = (m, pt_a[1] - m * pt_a[0], False)
            else:
                coefs[lane_id] = (pt_a[0], 0, True)
        return coefs

    def _evaluate_line_y(self, lane_id, cx):
        m, c, is_vertical = self.lane_coefficients[lane_id]
        return m if is_vertical else m * cx + c

    def _raw_zone(self, cx, cy):
        if cv2.pointPolygonTest(self.detection_polygon, (float(cx), float(cy)), False) < 0:
            return "Above" if cy < TOP_Y else "Below"
        if cy < self._evaluate_line_y("L1", cx):
            return "L"
        if cy < self._evaluate_line_y("L2", cx):
            return "M"
        return "R"

    def _debounced_zone(self, pid, raw_zone):
        if raw_zone not in ("L", "M", "R"):
            return raw_zone
        streak = self.ped_lane_streak[pid]
        streak.append(raw_zone)
        if len(streak) < LANE_DEBOUNCE_FRAMES:
            return raw_zone
        counts = {}
        for z in streak:
            counts[z] = counts.get(z, 0) + 1
        return max(counts, key=counts.get)

    def _get_zone(self, pid, cx, cy):
        return self._debounced_zone(pid, self._raw_zone(cx, cy))

    def _get_veh_lane_only(self, cx, cy):
        if cv2.pointPolygonTest(self.detection_polygon, (float(cx), float(cy)), False) < 0:
            return "Approach"
        return self._raw_zone(cx, cy)

    def _compute_grid(self, num_meters=45):
        lines = []
        for m in range(num_meters + 1):
            a = 1 - m / num_meters
            lines.append({
                "m": m,
                "top": interp(TOP_LEFT, TOP_RIGHT, a),
                "bottom": interp(BOTTOM_LEFT, BOTTOM_RIGHT, a),
            })
        return lines

    def _get_meter_from_x(self, x_front):
        return min(self.grid, key=lambda g: abs(g["top"][0] - x_front))["m"]

    def _process_pipeline(self, frame):
        results = self.model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
            device=self.device,
            imgsz=YOLO_INTERNAL_RESOLUTION,
        )[0]
        all_detections = sv.Detections.from_ultralytics(results)

        filtered_detections = all_detections[self.aoi_zone.trigger(all_detections)]
        if len(filtered_detections) == 0:
            return sv.Detections.empty(), sv.Detections.empty()

        tracked_detections = self.tracker.update_with_detections(filtered_detections)
        if len(tracked_detections) == 0:
            return sv.Detections.empty(), sv.Detections.empty()

        veh_mask = tracked_detections.class_id == 0
        ped_mask = (tracked_detections.class_id == 1) | (tracked_detections.class_id == 2)

        peds = tracked_detections[ped_mask]
        vehs = tracked_detections[veh_mask]

        if len(peds) > 0:
            boxes = peds.xyxy
            crop_mask = (
                (boxes[:, 0] >= CROP_X_MIN)
                & (boxes[:, 1] >= CROP_Y_MIN)
                & (boxes[:, 2] <= CROP_X_MAX)
                & (boxes[:, 3] <= CROP_Y_MAX)
            )
            peds = peds[crop_mask]

        return peds, vehs

    def _compute_multi_ped_gaps(self, vehs, current_peds_map, timestamp):
        for v_idx in range(len(vehs)):
            vid = vehs.tracker_id[v_idx]
            xyxy = vehs.xyxy[v_idx]
            front_x, _ = self._veh_front_anchor(xyxy)
            front_m = self._get_meter_from_x(front_x)
            speed = self.veh_speed_history.get(vid, [])[-1] if self.veh_speed_history.get(vid) else 0.0

            for pid, (ped_m, ped_status, ped_anchor) in current_peds_map.items():
                pair_key = (vid, pid)
                short_status = _get_short_status_code(ped_status, ped_anchor)
                self.pair_status_history.setdefault(pair_key, set()).add(short_status)
                dist = front_m - ped_m

                if speed > 0.5:
                    if front_m > ped_m:
                        tta = dist / speed
                        if tta > self.pair_max_tta.get(pair_key, -float("inf")):
                            self.pair_max_tta[pair_key] = tta
                            self.pair_min_dist[pair_key] = dist
                elif speed <= 0.5 and front_m > ped_m:
                    if pair_key not in self.pair_max_tta:
                        self.pair_max_tta[pair_key] = 0.0
                        self.pair_min_dist[pair_key] = dist

                if front_m <= ped_m and pair_key in self.pair_max_tta and pair_key not in self.pair_crossing_time:
                    self.pair_crossing_time[pair_key] = timestamp
                    self._log_interaction(vid, pid, timestamp)

    def _log_interaction(self, vid, pid, t1):
        self.finalized_interactions.append({
            "vid": vid,
            "pid": pid,
            "t1": t1,
            "d_gap_at_tta": self.pair_min_dist.get((vid, pid), 0.0),
            "lane_t1": self.veh_current_lane.get(vid, "N/A"),
        })

    def _track_segment_speed(self, vehs, timestamp):
        for i in range(len(vehs)):
            vid = vehs.tracker_id[i]
            x1, y1, x2, y2 = map(int, vehs.xyxy[i])
            front_m_band = self._get_meter_from_x(x2)
            prev_m_band = self.veh_last_m_band_crossed.get(vid, front_m_band)

            crossed_ref = False
            if prev_m_band > HEADWAY_REFERENCE_METER and front_m_band <= HEADWAY_REFERENCE_METER:
                crossed_ref = True
            elif front_m_band <= HEADWAY_REFERENCE_METER and vid not in self.veh_time_at_ref_line:
                crossed_ref = True

            if crossed_ref:
                self.veh_time_at_ref_line[vid] = timestamp
                if self.last_global_ref_time is not None:
                    self.live_veh_gaps[vid] = timestamp - self.last_global_ref_time
                else:
                    self.live_veh_gaps[vid] = 0.0
                self.last_global_ref_time = timestamp

            if front_m_band != prev_m_band and front_m_band < prev_m_band:
                prev_line_x = [g["top"][0] for g in self.grid if g["m"] == prev_m_band][0]
                if x2 > prev_line_x:
                    if (vid, prev_m_band) not in self.veh_segment_cross_time:
                        self.veh_segment_cross_time[(vid, prev_m_band)] = timestamp
                    self.veh_last_m_band_crossed[vid] = front_m_band

            if vid not in self.veh_last_m_band_crossed:
                self.veh_last_m_band_crossed[vid] = front_m_band
                self.veh_segment_cross_time[(vid, front_m_band)] = timestamp

    def _is_waiting_location(self, cx, cy, zone, direction, before_crossing=True):
        if _is_in_poly(ABOVE_WAITING_POLY, cx, cy) or _is_in_poly(BELOW_WAITING_POLY, cx, cy):
            return True
        if not before_crossing:
            return False
        if direction == "Up" and zone in ("R", "Below"):
            return True
        if direction == "Down" and zone in ("Above",):
            return True
        return False

    @staticmethod
    def _is_stationary_xy(x0, y0, x1, y1, dt):
        if dt <= 0:
            return False
        vx = abs(x1 - x0) / dt
        vy = abs(y1 - y0) / dt
        return vx <= WAIT_STATIONARY_PX_PER_S and vy <= WAIT_STATIONARY_PX_PER_S

    def _counts_as_waiting_interval(self, x0, y0, x1, y1, z1, direction, dt, before_crossing=True):
        """Wait time: in a waiting area OR nearly stationary in x and y."""
        if not before_crossing:
            return False
        in_area = self._is_waiting_location(x1, y1, z1, direction, before_crossing=True)
        stationary = self._is_stationary_xy(x0, y0, x1, y1, dt)
        return in_area or stationary

    def _infer_direction(self, df):
        direction = "Down"
        l_entry = df.loc[df.zone == "L", "t"].min()
        m_entry = df.loc[df.zone == "M", "t"].min()
        r_entry = df.loc[df.zone == "R", "t"].min()

        valid_entries = []
        if pd.notna(l_entry):
            valid_entries.append((l_entry, "L"))
        if pd.notna(m_entry):
            valid_entries.append((m_entry, "M"))
        if pd.notna(r_entry):
            valid_entries.append((r_entry, "R"))
        valid_entries.sort(key=lambda x: x[0])

        if len(valid_entries) >= 2:
            first, last = valid_entries[0][1], valid_entries[-1][1]
            if (first in ["R", "M"]) and last == "L":
                direction = "Up"
            elif first == "R" and last == "M":
                direction = "Up"
        elif len(valid_entries) == 0:
            # No lane crossings yet — infer from waiting-area presence
            wb_hits = sum(
                1 for _, cx, cy, _ in df.values if _is_in_poly(BELOW_WAITING_POLY, cx, cy)
            )
            wu_hits = sum(
                1 for _, cx, cy, _ in df.values if _is_in_poly(ABOVE_WAITING_POLY, cx, cy)
            )
            below_hits = (df.zone == "Below").sum()
            above_hits = (df.zone == "Above").sum()
            if wb_hits + below_hits > wu_hits + above_hits:
                direction = "Up"
        return direction, l_entry, m_entry, r_entry

    def _stable_cross_start(self, pid, direction, l_entry, m_entry, r_entry):
        streak = self.ped_lane_streak.get(pid, deque())
        if len(streak) < LANE_DEBOUNCE_FRAMES:
            if direction == "Down":
                return l_entry if pd.notna(l_entry) else m_entry
            return m_entry if pd.notna(m_entry) else r_entry

        if direction == "Down":
            return l_entry if pd.notna(l_entry) else m_entry
        return m_entry if pd.notna(m_entry) else r_entry

    def _compute_waiting_time_from_history(self, pid, direction, cross_start=None):
        hist = self.ped_history.get(pid, [])
        if len(hist) < 2:
            return 0.0

        if cross_start is None or (isinstance(cross_start, float) and pd.isna(cross_start)):
            end_t = float("inf")
        else:
            end_t = cross_start

        total = 0.0
        for i in range(1, len(hist)):
            t0, x0, y0, z0 = hist[i - 1]
            t1, x1, y1, z1 = hist[i]
            if t1 > end_t:
                break
            dt = t1 - t0
            if dt <= 0:
                continue
            if self._counts_as_waiting_interval(x0, y0, x1, y1, z1, direction, dt, before_crossing=True):
                total += dt
        return total

    def _calculate_full_ped_metrics(self, pid):
        df = pd.DataFrame(self.ped_history.get(pid, []), columns=["t", "cx", "cy", "zone"])
        if df.empty or df[df.zone.isin(["L", "M", "R"])].shape[0] < 1:
            return None

        past_lanes = {z for z in df["zone"] if z in ["L", "M", "R"]}
        if len(past_lanes) < 2:
            return None

        direction, l_entry, m_entry, r_entry = self._infer_direction(df)
        lane_entries = df.loc[df.zone.isin(["L", "M", "R"]), "t"]
        cross_end = lane_entries.max()

        t_exit_l, t_exit_m, t_exit_r = None, None, None
        timeline = []

        if direction == "Down":
            t_exit_l = m_entry if pd.notna(m_entry) else (r_entry if pd.notna(r_entry) else cross_end)
            t_exit_m = r_entry if pd.notna(r_entry) else cross_end
            t_exit_r = cross_end
            if pd.notna(l_entry):
                timeline.append(("L", l_entry, t_exit_l))
            if pd.notna(m_entry):
                timeline.append(("M", m_entry, t_exit_m))
            if pd.notna(r_entry):
                timeline.append(("R", r_entry, t_exit_r))
        else:
            t_exit_r = m_entry if pd.notna(m_entry) else (l_entry if pd.notna(l_entry) else cross_end)
            t_exit_m = l_entry if pd.notna(l_entry) else cross_end
            t_exit_l = cross_end
            if pd.notna(r_entry):
                timeline.append(("R", r_entry, t_exit_r))
            if pd.notna(m_entry):
                timeline.append(("M", m_entry, t_exit_m))
            if pd.notna(l_entry):
                timeline.append(("L", l_entry, t_exit_l))

        cross_start = self._stable_cross_start(pid, direction, l_entry, m_entry, r_entry)
        wait_time = self._compute_waiting_time_from_history(pid, direction, cross_start)

        def calc_speed(entry, exit_t):
            if pd.isna(entry) or pd.isna(exit_t):
                return None
            duration = exit_t - entry
            if duration <= 0.001:
                return None
            raw_speed = LANE_WIDTH / duration
            return min(raw_speed, 4.5) if duration < 0.25 else raw_speed

        measured_lanes = len({
            row_zone
            for t_val, _, _, row_zone in df.values
            if row_zone in ["L", "M", "R"] and (cross_start <= t_val <= cross_end)
        })
        active_distance = measured_lanes * LANE_WIDTH
        active_duration = cross_end - cross_start
        overall_spd = (active_distance / active_duration) if active_duration > 0.001 else 0.0

        return {
            "pid": pid,
            "direction": direction,
            "T_start_crossing": lane_entries.min(),
            "frozen_wait_time": wait_time,
            "cross_end": cross_end,
            "cross_start": cross_start,
            "T_exit_L": t_exit_l,
            "T_exit_M": t_exit_m,
            "T_exit_R": t_exit_r,
            "speed_L": calc_speed(l_entry, t_exit_l),
            "speed_M": calc_speed(m_entry, t_exit_m),
            "speed_R": calc_speed(r_entry, t_exit_r),
            "overall_speed": overall_spd,
            "timeline": timeline,
        }

    def generate_pedestrian_reports(self):
        reports = {}
        unique_pids = sorted({d["pid"] for d in self.finalized_interactions})
        speed_map = self.compute_vehicle_speed_metrics().set_index("ID")["Average_Speed"].to_dict()

        for target_pid in unique_pids:
            ped_v = self._calculate_full_ped_metrics(target_pid)
            if not ped_v:
                continue

            rows = []
            ped_dir = ped_v.get("direction", "Down")
            finish_time = ped_v.get("T_exit_R" if ped_dir == "Down" else "T_exit_L") or ped_v.get("cross_end", 99999.0)

            ped_interactions = sorted(
                [d for d in self.finalized_interactions if d["pid"] == target_pid],
                key=lambda x: x["t1"],
            )
            filtered_interactions = [
                d for d in ped_interactions
                if not (isinstance(finish_time, (int, float)) and self.veh_first_seen.get(d["vid"], 0.0) > finish_time)
            ]

            valid_rows_data = []
            has_accepted_gap = False

            for d in filtered_interactions:
                vid = d["vid"]
                veh_t0 = self.veh_first_seen.get(vid, 0.0)
                veh_t1 = d["t1"]
                d_gap_display = d["d_gap_at_tta"]

                real_status = set()
                start_cross = ped_v.get("T_start_crossing")
                if isinstance(start_cross, (int, float)) and veh_t0 < start_cross:
                    real_status.add("WU" if ped_dir == "Down" else "WB")
                for lane, l_start, l_end in ped_v.get("timeline", []):
                    if max(veh_t0, l_start) < min(veh_t1, l_end):
                        real_status.add(lane)
                if isinstance(ped_v.get("cross_end"), (int, float)) and veh_t1 > ped_v.get("cross_end"):
                    real_status.add("X")

                ped_status_display = ", ".join(sort_ped_statuses(list(real_status), direction=ped_dir))
                decision = "Rejected"

                if bool(real_status.intersection({"L", "M", "R", "X"})) and d["lane_t1"] in ["L", "M", "R"]:
                    lane_exit_time = ped_v.get(f"T_exit_{d['lane_t1']}")
                    if isinstance(veh_t1, (int, float)) and isinstance(lane_exit_time, (int, float)) and lane_exit_time < veh_t1:
                        decision = "Accepted"

                if isinstance(finish_time, (int, float)) and isinstance(veh_t1, (int, float)) and veh_t1 > finish_time:
                    d_gap_display = 36.00
                    decision = "Accepted"

                row_data = {
                    "Video_Name": self.video_name,
                    "Veh_ID": vid,
                    "Ped_ID": target_pid,
                    "Veh_T1_Time_Val": d["t1"],
                    "Veh_T1_Time": f"{d['t1']:.2f}",
                    "Ped_Start_Time": f"{ped_v.get('T_start_crossing', 'N/A'):.2f}"
                    if isinstance(ped_v.get("T_start_crossing"), (int, float))
                    else "N/A",
                    "Ped_Direction": ped_dir,
                    "Ped_Decision": decision,
                    "Veh_Dist_Gap": f"{d_gap_display:.2f}" if isinstance(d_gap_display, (int, float)) else d_gap_display,
                    "Veh_Lane": d["lane_t1"],
                    "Ped_Status": ped_status_display,
                    "Veh_Speed": f"{speed_map.get(vid, 0.0):.2f}" if isinstance(speed_map.get(vid), (int, float)) else "N/A",
                    "Ped_Wait_Time": f"{ped_v.get('frozen_wait_time', 0.0):.2f}"
                    if isinstance(ped_v.get("frozen_wait_time"), (int, float))
                    else "N/A",
                    "Ped_Avg_Speed": f"{ped_v.get('overall_speed', 0.0):.2f}"
                    if isinstance(ped_v.get("overall_speed"), (int, float))
                    else "N/A",
                    "Ped_Speed_L": f"{ped_v.get('speed_L', 0.0):.2f}" if isinstance(ped_v.get("speed_L"), (int, float)) else "N/A",
                    "Ped_Speed_M": f"{ped_v.get('speed_M', 0.0):.2f}" if isinstance(ped_v.get("speed_M"), (int, float)) else "N/A",
                    "Ped_Speed_R": f"{ped_v.get('speed_R', 0.0):.2f}" if isinstance(ped_v.get("speed_R"), (int, float)) else "N/A",
                }

                if decision == "Rejected":
                    valid_rows_data.append(row_data)
                elif decision == "Accepted" and not has_accepted_gap:
                    valid_rows_data.append(row_data)
                    has_accepted_gap = True

            valid_rows_data.sort(key=lambda x: x["Veh_T1_Time_Val"])
            previous_t1 = None
            for r in valid_rows_data:
                current_t1 = r["Veh_T1_Time_Val"]
                r["Veh_Headway"] = (
                    f"{current_t1 - previous_t1:.2f}"
                    if previous_t1 is not None
                    else f"{self.live_veh_gaps.get(r['Veh_ID'], 0.0):.2f}"
                )
                previous_t1 = current_t1
                del r["Veh_T1_Time_Val"]
                rows.append(r)

            if rows:
                reports[target_pid] = pd.DataFrame(rows)

        return reports

    def compute_vehicle_speed_metrics(self):
        rows = []
        for vid in {d["vid"] for d in self.finalized_interactions}:
            speeds = self.veh_speed_history.get(vid, [])
            rows.append({"ID": vid, "Average_Speed": np.mean(speeds) if speeds else "N/A"})
        return pd.DataFrame(rows)

    def _update_analytics(self, peds, vehs, timestamp):
        current_peds_map = {}

        for i in range(len(peds)):
            pid = peds.tracker_id[i]
            cid = peds.class_id[i]
            cx, cy = self._ped_anchor(peds.xyxy[i])
            zone = self._get_zone(pid, cx, cy)
            self.ped_history.setdefault(pid, []).append((timestamp, cx, cy, zone))

            streak = self.ped_lane_streak[pid]
            if zone in ("L", "M", "R") and len(streak) >= LANE_DEBOUNCE_FRAMES and len(set(streak)) == 1:
                if not self.is_crossing_started.get(pid, False):
                    self.is_crossing_started[pid] = True

            if zone in ("L", "M", "R"):
                status = f"Crossing ({zone})"
            elif self.is_crossing_started.get(pid, False):
                past_zones = {h[3] for h in self.ped_history[pid] if h[3] in ("L", "M", "R")}
                status = "Crossed" if len(past_zones) >= 2 else f"Waiting ({zone})"
            else:
                status = f"Waiting ({zone})"

            self.ped_overall_status[pid] = status
            ped_m = self._get_meter_from_x(cx)
            current_peds_map[pid] = (ped_m, status, (cx, cy))

            hist_df = pd.DataFrame(self.ped_history[pid], columns=["t", "cx", "cy", "zone"])
            direction, l_entry, m_entry, r_entry = self._infer_direction(hist_df)
            cross_start = None
            if self.is_crossing_started.get(pid):
                cross_start = self._stable_cross_start(pid, direction, l_entry, m_entry, r_entry)
            self.ped_live_wait_time[pid] = self._compute_waiting_time_from_history(
                pid, direction, cross_start
            )

        self._compute_multi_ped_gaps(vehs, current_peds_map, timestamp)
        self._track_segment_speed(vehs, timestamp)

        for i in range(len(vehs)):
            vid = vehs.tracker_id[i]
            front_x, front_y = self._veh_front_anchor(vehs.xyxy[i])
            lane_str = self._get_veh_lane_only(front_x, front_y)
            self.veh_current_lane[vid] = lane_str

            curr_m = self._get_meter_from_x(front_x)
            if vid not in self.veh_first_seen:
                self.veh_first_seen[vid] = timestamp
                self.last_meter_band[vid] = curr_m
                self.last_meter_time[vid] = timestamp

            prev_m, prev_t = self.last_meter_band.get(vid), self.last_meter_time.get(vid)
            if prev_m is not None and (timestamp - prev_t) > 0 and curr_m != prev_m:
                speed = abs(curr_m - prev_m) / (timestamp - prev_t)
                if speed > 0:
                    self.veh_speed_history.setdefault(vid, []).append(speed)
                self.last_meter_band[vid] = curr_m
                self.last_meter_time[vid] = timestamp

    def _draw_ped_text(self, frame, x1, y1, x2, y2, status_line, wait_line, color):
        font, scale, thickness = 0, 0.5, 2
        (_, h_status), _ = cv2.getTextSize(status_line, font, scale, thickness)
        (_, h_wait), _ = cv2.getTextSize(wait_line, font, scale, thickness)
        line_gap = 3
        block_h = h_status + line_gap + h_wait + 6

        if y1 - block_h >= 0:
            y_wait = y1 - 6
            y_status = y_wait - line_gap - h_status
        else:
            y_status = y2 + 6 + h_status
            y_wait = y_status + line_gap

        x_text = max(2, min(x1, FRAME_WIDTH - 120))
        cv2.putText(frame, status_line, (x_text, y_status), font, scale, color, thickness, cv2.LINE_AA)
        cv2.putText(frame, wait_line, (x_text, y_wait), font, scale, (200, 255, 255), thickness, cv2.LINE_AA)

    def _draw(self, frame, peds, vehs, timestamp):
        overlay = frame.copy()
        cv2.polylines(frame, [self.detection_polygon], True, (255, 0, 0), 2)
        cv2.polylines(frame, [self.aoi_polygon_coords], True, (255, 255, 0), 1)

        for name, line_pts in self.lane_lines.items():
            cv2.line(frame, line_pts[0], line_pts[1], (255, 0, 150), 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, name, (line_pts[0][0] + 10, line_pts[0][1] - 10), 0, 0.4, (255, 0, 150), 1)

        cv2.fillPoly(overlay, [ABOVE_WAITING_POLY], (0, 150, 255))
        cv2.fillPoly(overlay, [BELOW_WAITING_POLY], (255, 150, 0))
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        for i in range(len(peds)):
            pid = peds.tracker_id[i]
            cid = peds.class_id[i]
            x1, y1, x2, y2 = map(int, peds.xyxy[i])
            cx, cy = self._ped_anchor(peds.xyxy[i])
            status = self.ped_overall_status.get(pid, "N/A")
            wait_s = self.ped_live_wait_time.get(pid, 0.0)

            prefix = "G" if cid == 2 else "P"
            if cid == 2:
                box_color = (255, 0, 255)
            else:
                box_color = (0, 0, 255) if "Crossing" in status else (0, 255, 0)

            status_line = f"{prefix}{pid}: {status}"
            wait_line = f"wait {wait_s:.1f}s"

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
            self._draw_ped_text(frame, x1, y1, x2, y2, status_line, wait_line, (255, 255, 255))

        for i in range(len(vehs)):
            vid = vehs.tracker_id[i]
            x1, y1, x2, y2 = map(int, vehs.xyxy[i])
            fx, fy = self._veh_front_anchor(vehs.xyxy[i])
            lane_str = self.veh_current_lane.get(vid, "N/A")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 150, 0), 2)
            cv2.circle(frame, (fx, fy), 4, (0, 255, 255), -1)
            cv2.putText(frame, f"V{vid} {lane_str}", (x1, y1 - 5), 0, 0.5, (255, 150, 0), 2)

        return frame

    def process_video(self):
        if not Path(self.video_path).exists():
            raise FileNotFoundError(f"Missing source file: {self.video_path}")

        cap = cv2.VideoCapture(str(self.video_path))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._init_tracker()
        frame_i = 0

        print(f"[INFO] Running pipeline on {self.device.upper()} @ {self.fps:.1f} fps...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            ts = frame_i / self.fps

            p, v = self._process_pipeline(frame)
            self._update_analytics(p, v, ts)

            if frame_i % FRAME_SKIP == 0:
                frame = self._draw(frame, p, v, ts)
                cv2.imshow("Multi-Ped Analytics", frame)
                if cv2.waitKey(1) == 27:
                    break

            frame_i += 1

        cap.release()
        cv2.destroyAllWindows()

        ped_reports = self.generate_pedestrian_reports()
        all_dfs = []
        for pid, df in ped_reports.items():
            print(f"\n========== METRICS FOR PEDESTRIAN P{pid} ==========")
            print(df.to_markdown(index=False))
            all_dfs.append(df)

        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            output_dir = PROJECT_ROOT / "z_output"
            output_dir.mkdir(exist_ok=True)
            csv_path = output_dir / f"{self.video_name}.csv"
            final_df.to_csv(csv_path, index=False)
            print(f"\n[INFO] Analytics complete. Report saved: {csv_path}")


if __name__ == "__main__":
    PedestrianAnalyzer(VIDEO_PATH).process_video()
