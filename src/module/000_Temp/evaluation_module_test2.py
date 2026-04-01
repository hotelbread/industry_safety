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
    action_label: str      # top-1 action 이름
    confidence: float      # top-1 confidence
    keypoints: Optional[np.ndarray] = None  # shape (17, 2) or None


@dataclass
class EvalResult:
    action_name: str           # 대상 동작 이름 (ex: "붐 올리기")
    conf_mean: float           # 신뢰도 평균
    conf_max: float            # 신뢰도 최대값
    first_detect_frame: int    # 처음으로 해당 동작 인식된 프레임 번호 (-1이면 미인식)
    detect_ratio: float        # 전체 프레임 중 인식 비율 (0~1)
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
    """

    def __init__(self):
        self._frames: list[FrameData] = []
        self._action_name = ""
        self._frame_counter = 0
        self._collecting = False

    # -------------------------------------------------------
    # 수집 제어
    # -------------------------------------------------------

    def start_collection(self, action_name: str):
        """새 동작 구간 수집 시작"""
        self._frames = []
        self._frame_counter = 0
        self._action_name = action_name
        self._collecting = True
        print(f"[EvalModule] 수집 시작: {action_name}")

    def stop_collection(self):
        """수집 중단 (evaluate() 호출 전에 사용 가능)"""
        self._collecting = False
        print(f"[EvalModule] 수집 중단. 총 {len(self._frames)} 프레임")

    def is_collecting(self) -> bool:
        return self._collecting

    def add_frame(
        self,
        action_label_results: list[dict],
        keypoints: Optional[np.ndarray] = None
    ):
        """
        프레임마다 호출.

        Parameters
        ----------
        action_label_results : list of dict
            AiThread에서 오는 형식: [{'label': str, 'conf': float}, ...]
            top-1 이 index 0.
        keypoints : np.ndarray, optional
            shape (17, 2) — active user의 keypoints
        """
        if not self._collecting:
            return

        if not action_label_results:
            # 이번 프레임에 인식 결과 없음 → 빈 프레임으로 기록
            self._frames.append(FrameData(
                frame_idx=self._frame_counter,
                action_label="",
                confidence=0.0,
                keypoints=keypoints
            ))
        else:
            top1 = action_label_results[0]
            self._frames.append(FrameData(
                frame_idx=self._frame_counter,
                action_label=top1.get("label", ""),
                confidence=top1.get("conf", 0.0),
                keypoints=keypoints
            ))

        self._frame_counter += 1

    # -------------------------------------------------------
    # 평가
    # -------------------------------------------------------

    def evaluate(self) -> EvalResult:
        """
        수집된 프레임 데이터로 EvalResult 반환.
        수집 중이더라도 현재까지 데이터로 계산.
        """
        self._collecting = False
        total_frames = len(self._frames)

        if total_frames == 0:
            return EvalResult(
                action_name=self._action_name,
                conf_mean=0.0,
                conf_max=0.0,
                first_detect_frame=-1,
                detect_ratio=0.0,
                angle_score=-1.0,
                total_score=0.0,
                total_frames=0
            )

        # ---- 1. action label 기반 ----
        target_action = self._action_name

        # 대상 동작이 인식된 프레임만 필터
        matched_frames = [
            f for f in self._frames
            if f.action_label == target_action
        ]

        confs = [f.confidence for f in matched_frames]
        conf_mean = float(np.mean(confs)) if confs else 0.0
        conf_max  = float(np.max(confs))  if confs else 0.0

        # 첫 인식 프레임
        first_detect_frame = matched_frames[0].frame_idx if matched_frames else -1

        # 인식 비율
        detect_ratio = len(matched_frames) / total_frames

        # ---- 2. keypoints 각도 기반 ----
        # 유효한 keypoints 프레임에서 평균 각도 점수 계산
        angle_scores = []
        for f in matched_frames:
            if f.keypoints is not None:
                s = calc_angle_score(target_action, f.keypoints)
                if s >= 0:
                    angle_scores.append(s)

        angle_score = float(np.mean(angle_scores)) if angle_scores else -1.0

        # ---- 3. 종합 점수 ----
        total_score = self._calc_total_score(
            conf_mean, conf_max, detect_ratio, angle_score
        )

        result = EvalResult(
            action_name=target_action,
            conf_mean=round(conf_mean * 100, 1),   # 0~100 %
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
        가중치 기반 종합 점수 계산 (0~100).

        가중치 (조정 가능):
          - detect_ratio : 40%  (얼마나 지속적으로 인식했나)
          - conf_mean    : 30%  (평균 신뢰도)
          - conf_max     : 10%  (최고 신뢰도)
          - angle_score  : 20%  (keypoints 각도, 없으면 나머지로 재분배)
        """
        # 각 항목을 0~100 스케일로 정규화
        s_ratio = detect_ratio * 100          # 이미 0~1 → 0~100
        s_mean  = conf_mean * 100             # 0~1 → 0~100
        s_max   = conf_max  * 100             # 0~1 → 0~100

        if angle_score >= 0:
            # keypoints 사용 가능
            score = (
                s_ratio * 0.40
                + s_mean  * 0.30
                + s_max   * 0.10
                + angle_score * 0.20
            )
        else:
            # keypoints 없음 → 나머지 항목으로 재분배
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
    각 단계별로 EvaluationModule 인스턴스를 유지.

    사용 예시:
        exp_eval = ExperienceEvaluator(["붐 올리기", "권상", "비상 정지"])
        exp_eval.start_stage(0)
        for frame:
            exp_eval.add_frame(action_results, keypoints)
        exp_eval.end_stage()
        exp_eval.start_stage(1)
        ...
        summary = exp_eval.get_summary()
    """

    def __init__(self, stage_names: list[str]):
        self.stage_names = stage_names
        self.stage_count = len(stage_names)
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
        self._evaluators[stage_idx].start_collection(self.stage_names[stage_idx])

    def add_frame(self, action_label_results: list[dict], keypoints=None):
        if self._current_stage < 0:
            return
        self._evaluators[self._current_stage].add_frame(action_label_results, keypoints)

    def end_stage(self) -> Optional[EvalResult]:
        if self._current_stage < 0:
            return None
        result = self._evaluators[self._current_stage].evaluate()
        self._results[self._current_stage] = result
        self._current_stage = -1
        return result

    def get_summary(self) -> list[Optional[EvalResult]]:
        """단계별 EvalResult 리스트 반환"""
        return self._results

    def get_total_score(self) -> float:
        """완료된 단계들의 총합 평균 점수"""
        valid = [r.total_score for r in self._results if r is not None]
        return round(sum(valid) / len(valid), 1) if valid else 0.0

    def reset(self):
        """전체 초기화"""
        self._evaluators = [EvaluationModule() for _ in self.stage_names]
        self._results = [None] * self.stage_count
        self._current_stage = -1
        print("[ExpEval] 초기화 완료")
