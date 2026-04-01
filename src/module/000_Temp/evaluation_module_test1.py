"""
evaluation_module.py

체험 동작 평가 모듈.
우선순위:
  1. action_label_results 기반 → 신뢰도 평균, 최대값, 첫 인식 프레임
  2. keypoints 기반 → 관절 각도 평가 (선택적)

사용법:
    evaluator = EvaluationModule()
    evaluator.start_collection(action_name="붐 올리기")
    for each frame:
        evaluator.add_frame(action_label_results, keypoints=None)
    result = evaluator.evaluate()
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import Optional


# -----------------------------------------------------------------
# 데이터 구조
# -----------------------------------------------------------------

@dataclass
class FrameData:
    frame_idx: int
    scores_array: Optional[np.ndarray]     # shape (num_classes,) — 전체 클래스 conf 배열
    keypoints: Optional[np.ndarray] = None # shape (17, 2) or None


@dataclass
class EvalResult:
    action_name: str           # 대상 동작 이름 (ex: "붐 올리기")
    target_class_idx: int      # 평가 대상 클래스 인덱스
    conf_mean: float           # 대상 클래스 신뢰도 평균 (0~100%)
    conf_max: float            # 대상 클래스 신뢰도 최대값 (0~100%)
    first_detect_frame: int    # 처음으로 conf > threshold 넘긴 프레임 번호 (-1이면 없음)
    detect_ratio: float        # 전체 프레임 중 threshold 초과 비율 (0~100%)
    angle_score: float         # 관절 각도 기반 점수 (0~100, keypoints 없으면 -1)
    total_score: float         # 종합 점수 (0~100)
    total_frames: int          # 수집된 총 프레임 수


# -----------------------------------------------------------------
# 각도 계산 유틸
# -----------------------------------------------------------------

def calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    세 점 a-b-c 에서 b를 꼭짓점으로 하는 각도(도) 반환.
    a, b, c: shape (2,) or (3,) — x, y 만 사용
    """
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    cos_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


# -----------------------------------------------------------------
# 동작별 기준 각도 정의
# -----------------------------------------------------------------
# keypoint index (COCO 17):
#   0:코, 5:왼어깨, 6:오른어깨, 7:왼팔꿈치, 8:오른팔꿈치,
#   9:왼손목, 10:오른손목, 11:왼엉덩이, 12:오른엉덩이,
#   13:왼무릎, 14:오른무릎, 15:왼발목, 16:오른발목

ACTION_ANGLE_CRITERIA = {
    "붐 올리기": [
        # (a_idx, b_idx, c_idx, target_angle, tolerance, weight)
        # 오른쪽: 오른어깨-오른팔꿈치-오른손목 → 팔을 위로 뻗을때 각도 크게
        (6, 8, 10, 160.0, 25.0, 1.0),
        # 왼쪽: 왼어깨-왼팔꿈치-왼손목
        (5, 7, 9, 160.0, 25.0, 1.0),
    ],
    "권상": [
        # 손목이 어깨 위에 위치하는 동작 → 어깨-팔꿈치-손목 각도
        (6, 8, 10, 150.0, 30.0, 1.0),
        (5, 7, 9,  150.0, 30.0, 1.0),
    ],
    "비상 정지": [
        # 양팔을 수평으로 벌리는 동작 → 어깨-팔꿈치가 수평
        # 어깨-팔꿈치-손목: 일자로 뻗음
        (6, 8, 10, 170.0, 20.0, 1.0),
        (5, 7, 9,  170.0, 20.0, 1.0),
    ],
}


def calc_angle_score(action_name: str, keypoints: np.ndarray) -> float:
    """
    동작 이름과 keypoints(17, 2)를 받아 각도 점수(0~100) 반환.
    기준 각도 정의가 없으면 -1 반환.
    """
    criteria = ACTION_ANGLE_CRITERIA.get(action_name)
    if criteria is None:
        return -1.0
    if keypoints is None or keypoints.shape[0] < 17:
        return -1.0

    scores = []
    weights = []

    for (a_idx, b_idx, c_idx, target, tolerance, weight) in criteria:
        angle = calc_angle(keypoints[a_idx], keypoints[b_idx], keypoints[c_idx])
        diff = abs(angle - target)
        # tolerance 이내면 100점, 벗어날수록 선형 감소, 2*tolerance 이상이면 0점
        point = max(0.0, 1.0 - diff / tolerance) * 100.0
        scores.append(point)
        weights.append(weight)

    total_weight = sum(weights)
    if total_weight < 1e-6:
        return -1.0

    return sum(s * w for s, w in zip(scores, weights)) / total_weight


# -----------------------------------------------------------------
# 메인 클래스
# -----------------------------------------------------------------

class EvaluationModule:
    """
    단일 동작 구간의 프레임 데이터를 수집하고 평가.

    add_frame에 넘기는 scores_array:
        np.ndarray shape (num_classes,) — ai_thread에서 받은 전체 클래스 conf 배열
        None이면 해당 프레임은 미검출로 기록
    """

    # conf가 이 값 이상이면 "해당 동작을 인식했다"고 판단
    DETECT_THRESHOLD = 0.3

    def __init__(self):
        self._frames: list[FrameData] = []
        self._action_name   = ""
        self._target_class_idx = -1   # 평가 대상 클래스 인덱스
        self._frame_counter = 0
        self._collecting    = False

    # -------------------------------------------------------
    # 수집 제어
    # -------------------------------------------------------

    def start_collection(self, action_name: str, target_class_idx: int):
        """
        새 동작 구간 수집 시작.

        Parameters
        ----------
        action_name      : 동작 이름 (표시용)
        target_class_idx : 평가할 클래스 인덱스 (label_map 기준)
                           예) Raise Boom=1, Raise Load=2, Emergency Stop=3
        """
        self._frames           = []
        self._frame_counter    = 0
        self._action_name      = action_name
        self._target_class_idx = target_class_idx
        self._collecting       = True
        print(f"[EvalModule] 수집 시작: {action_name} (class_idx={target_class_idx})")

    def stop_collection(self):
        self._collecting = False
        print(f"[EvalModule] 수집 중단. 총 {len(self._frames)} 프레임")

    def is_collecting(self) -> bool:
        return self._collecting

    def add_frame(
        self,
        scores_array: Optional[np.ndarray],
        keypoints: Optional[np.ndarray] = None
    ):
        """
        프레임마다 호출.

        Parameters
        ----------
        scores_array : np.ndarray shape (num_classes,) 또는 None
            ai_thread에서 받은 전체 클래스 conf 배열.
            None이면 해당 프레임은 미검출로 처리.
        keypoints    : np.ndarray shape (17, 2) 또는 None
        """
        if not self._collecting:
            return

        # ndarray 여부 확인 후 저장
        if scores_array is not None and not isinstance(scores_array, np.ndarray):
            scores_array = None   # 잘못된 타입이면 None으로 처리

        self._frames.append(FrameData(
            frame_idx=self._frame_counter,
            scores_array=scores_array,
            keypoints=keypoints
        ))
        self._frame_counter += 1

    # -------------------------------------------------------
    # 평가
    # -------------------------------------------------------

    def evaluate(self) -> EvalResult:
        """
        수집된 프레임 데이터로 EvalResult 반환.
        """
        self._collecting = False
        total_frames = len(self._frames)
        idx = self._target_class_idx

        # 빈 결과 반환 조건
        if total_frames == 0 or idx < 0:
            return EvalResult(
                action_name=self._action_name,
                target_class_idx=idx,
                conf_mean=0.0,
                conf_max=0.0,
                first_detect_frame=-1,
                detect_ratio=0.0,
                angle_score=-1.0,
                total_score=0.0,
                total_frames=0
            )

        # ---- 1. 대상 클래스 conf 추출 ----
        confs = []
        for f in self._frames:
            if f.scores_array is not None and idx < len(f.scores_array):
                confs.append(float(f.scores_array[idx]))
            else:
                confs.append(0.0)

        confs_arr = np.array(confs)

        conf_mean = float(np.mean(confs_arr))
        conf_max  = float(np.max(confs_arr))

        # threshold 초과 프레임 — "해당 동작을 인식한 프레임"
        detected_mask  = confs_arr >= self.DETECT_THRESHOLD
        detected_count = int(np.sum(detected_mask))
        detect_ratio   = detected_count / total_frames

        # 첫 인식 프레임
        detected_indices = np.where(detected_mask)[0]
        first_detect_frame = int(detected_indices[0]) if len(detected_indices) > 0 else -1

        # ---- 2. keypoints 각도 기반 ----
        angle_scores = []
        for i, f in enumerate(self._frames):
            if detected_mask[i] and f.keypoints is not None:
                s = calc_angle_score(self._action_name, f.keypoints)
                if s >= 0:
                    angle_scores.append(s)

        angle_score = float(np.mean(angle_scores)) if angle_scores else -1.0

        # ---- 3. 종합 점수 ----
        total_score = self._calc_total_score(
            conf_mean, conf_max, detect_ratio, angle_score
        )

        result = EvalResult(
            action_name=self._action_name,
            target_class_idx=idx,
            conf_mean=round(conf_mean * 100, 1),
            conf_max=round(conf_max * 100, 1),
            first_detect_frame=first_detect_frame,
            detect_ratio=round(detect_ratio * 100, 1),
            angle_score=round(angle_score, 1),
            total_score=round(total_score, 1),
            total_frames=total_frames
        )

        print(f"[EvalModule] 평가 완료: {result}")
        return result

    def _calc_total_score(
        self,
        conf_mean: float,
        conf_max: float,
        detect_ratio: float,
        angle_score: float
    ) -> float:
        """
        가중치 기반 종합 점수 (0~100).
          - detect_ratio : 40%  (얼마나 지속적으로 인식했나)
          - conf_mean    : 30%  (평균 신뢰도)
          - conf_max     : 10%  (최고 신뢰도)
          - angle_score  : 20%  (keypoints 각도, 없으면 나머지로 재분배)
        """
        s_ratio = detect_ratio * 100
        s_mean  = conf_mean    * 100
        s_max   = conf_max     * 100

        if angle_score >= 0:
            score = (
                s_ratio * 0.40
                + s_mean  * 0.30
                + s_max   * 0.10
                + angle_score * 0.20
            )
        else:
            score = (
                s_ratio * 0.50
                + s_mean  * 0.35
                + s_max   * 0.15
            )

        return min(100.0, max(0.0, score))


# -----------------------------------------------------------------
# 다중 동작 결과를 합산하는 래퍼 (mode_3 호출용)
# -----------------------------------------------------------------

class ExperienceEvaluator:
    """
    3단계 체험 전체를 관리하는 평가 래퍼.

    사용 예시:
        exp_eval = ExperienceEvaluator(
            stage_names=["Raise Boom", "Raise Load", "Emergency Stop"],
            stage_class_idxs=[1, 2, 3]
        )
        exp_eval.start_stage(0)
        for frame:
            scores_array = action_results.get(active_user_id)  # ndarray or None
            exp_eval.add_frame(scores_array, keypoints=None)
        exp_eval.end_stage()
        ...
        summary = exp_eval.get_summary()
    """

    def __init__(self, stage_names: list[str], stage_class_idxs: list[int]):
        assert len(stage_names) == len(stage_class_idxs), \
            "stage_names와 stage_class_idxs 길이가 같아야 합니다."
        self.stage_names      = stage_names
        self.stage_class_idxs = stage_class_idxs
        self.stage_count      = len(stage_names)
        self._evaluators: list[EvaluationModule] = [
            EvaluationModule() for _ in stage_names
        ]
        self._results: list[Optional[EvalResult]] = [None] * self.stage_count
        self._current_stage = -1

    def start_stage(self, stage_idx: int):
        if stage_idx < 0 or stage_idx >= self.stage_count:
            print(f"[ExpEval] 잘못된 stage_idx: {stage_idx}")
            return
        self._current_stage = stage_idx
        self._evaluators[stage_idx].start_collection(
            action_name=self.stage_names[stage_idx],
            target_class_idx=self.stage_class_idxs[stage_idx]
        )

    def add_frame(
        self,
        scores_array: Optional[np.ndarray],
        keypoints: Optional[np.ndarray] = None
    ):
        """
        Parameters
        ----------
        scores_array : np.ndarray shape (num_classes,) 또는 None
            action_results.get(active_user_id) 로 꺼낸 값을 그대로 넘기면 됨
        keypoints    : np.ndarray shape (17, 2) 또는 None
        """
        if self._current_stage < 0:
            return
        self._evaluators[self._current_stage].add_frame(scores_array, keypoints)

    def end_stage(self) -> Optional[EvalResult]:
        if self._current_stage < 0:
            return None
        result = self._evaluators[self._current_stage].evaluate()
        self._results[self._current_stage] = result
        self._current_stage = -1
        return result

    def get_summary(self) -> list[Optional[EvalResult]]:
        return self._results

    def get_total_score(self) -> float:
        valid = [r.total_score for r in self._results if r is not None]
        return round(sum(valid) / len(valid), 1) if valid else 0.0

    def reset(self):
        self._evaluators = [EvaluationModule() for _ in self.stage_names]
        self._results    = [None] * self.stage_count
        self._current_stage = -1
        print("[ExpEval] 초기화 완료")
