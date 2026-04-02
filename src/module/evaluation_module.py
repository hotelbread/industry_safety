"""
evaluation_module.py  v2.0

변경사항:
  - 프레임마다 add_frame 하는 방식 → 구간 완료 후 analyze() 한 번에 처리
  - action_results 형식 변경: list[dict] → np.ndarray(num_classes,)
  - keypoints 133개 whole body 지원
  - 동작별 pose 각도 기준 업데이트 (손가락 포함 가능)

사용법 (main.py):
    # __init__
    self.exp_evaluator = ExperienceEvaluator(
        # stage_names      = ["Raise Boom", "Raise Load", "Emergency Stop"],
        stage_names = {
            0: "Raise Boom",
            1: "Raise Load",
            2: "Emergency Stop"}
        stage_class_idxs = [1, 2, 3],
    )

    # 동작 구간 시작 (action_recognition=True 시점)
    # self.exp_evaluator.start_stage(0)
    동작구간 시작을 알 필요없이 main에서 add_frame하면 그걸 쌓아서 한번에 추후 처리하는 구조로 수정

    # 매 프레임 (update_controller 안에서)
    scores = result_dict.get("action_results", {}).get(self.active_user_id)
    kpts   = result_dict.get("keypoints", {}).get(self.active_user_id, [])[0]
    self.exp_evaluator.add_frame(scores, kpts)

    # 구간 종료 (countdown_off 시점)
    result = self.exp_evaluator.end_stage()   # EvalResult 반환
    self.update_scoretable(result, row=0)
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Optional


# ================================================================
# 데이터 구조
# ================================================================

@dataclass
class EvalResult:
    stage_idx        : int    # 단계 번호 (0, 1, 2)
    action_name      : str    # 표시용 동작 이름
    target_class_idx : int    # 평가 대상 클래스 인덱스

    # ── action conf 기반 ──────────────────────────────────────────
    conf_mean         : float  # 대상 클래스 평균 conf (0~100%)
    conf_max          : float  # 대상 클래스 최대 conf (0~100%)
    detect_ratio      : float  # threshold 초과 프레임 비율 (0~100%)
    first_detect_frame: int    # 처음 threshold 초과한 프레임 번호 (-1이면 없음)

    # ── pose 각도 기반 ────────────────────────────────────────────
    angle_score: float  # 0~100, keypoints 없으면 -1.0

    # ── 종합 ──────────────────────────────────────────────────────
    total_score : float
    total_frames: int


# ================================================================
# 관절 각도 계산 유틸
# ================================================================

def calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """b를 꼭짓점으로 하는 a-b-c 각도(도). a,b,c shape: (2,) 이상"""
    ba = a[:2].astype(float) - b[:2].astype(float)
    bc = c[:2].astype(float) - b[:2].astype(float)
    n_ba, n_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if n_ba < 1e-6 or n_bc < 1e-6:
        return 0.0
    cos_a = np.clip(np.dot(ba, bc) / (n_ba * n_bc), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def calc_position_score(action_name: str, kpts: np.ndarray) -> float:
    """
    좌표 위치 기반 보조 점수 (0~100).
    각도만으로 잡기 어려운 수직/수평 위치관계 체크.
    kpts: shape (133, 2) — y축은 아래가 큰 값
    """
    if kpts is None or kpts.shape[0] < 17:
        return -1.0

    if action_name == "Raise Boom":
        # 양 손목 y ≈ 양 어깨 y (수평)
        r_diff  = abs(float(kpts[10][1]) - float(kpts[6][1]))
        l_diff  = abs(float(kpts[9][1])  - float(kpts[5][1]))
        r_score = max(0.0, 1.0 - r_diff / 100.0) * 100
        l_score = max(0.0, 1.0 - l_diff / 100.0) * 100
        return (r_score + l_score) / 2

    elif action_name == "Raise Load":
        # 손목이 어깨보다 위에 있어야 함 (y값이 작아야)
        r_up    = float(kpts[6][1]) - float(kpts[10][1])
        l_up    = float(kpts[5][1]) - float(kpts[9][1])
        r_score = min(100.0, max(0.0, r_up / 1.5))
        l_score = min(100.0, max(0.0, l_up / 1.5))
        return (r_score + l_score) / 2

    elif action_name == "Emergency Stop":
        # 양 손목이 어깨와 같은 높이 (수평)
        r_diff  = abs(float(kpts[10][1]) - float(kpts[6][1]))
        l_diff  = abs(float(kpts[9][1])  - float(kpts[5][1]))
        r_score = max(0.0, 1.0 - r_diff / 80.0) * 100
        l_score = max(0.0, 1.0 - l_diff / 80.0) * 100
        return (r_score + l_score) / 2

    return -1.0


# ================================================================
# 동작별 각도 기준 (whole body 133 keypoints)
# ================================================================
# COCO body 17:
#   0코  5왼어깨  6오른어깨  7왼팔꿈치  8오른팔꿈치
#   9왼손목  10오른손목  11왼엉덩이  12오른엉덩이
# ================================================================

ACTION_ANGLE_CRITERIA = {
    "Raise Boom": [
        # 팔이 수평으로 뻗음 — 어깨-팔꿈치-손목 일직선
        (6, 8, 10, 175.0, 20.0, 1.0),   # 오른팔
        (5, 7,  9, 175.0, 20.0, 1.0),   # 왼팔
        # 몸통 기준 수평 — 엉덩이-어깨-손목 약 90도
        (12, 6, 10, 90.0, 20.0, 0.8),
        (11, 5,  9, 90.0, 20.0, 0.8),
    ],
    "Raise Load": [
        # 팔이 수직으로 위 — 어깨-팔꿈치-손목 일직선
        (6, 8, 10, 175.0, 20.0, 1.0),
        (5, 7,  9, 175.0, 20.0, 1.0),
    ],
    "Emergency Stop": [
        # 양팔 완전히 수평으로 뻗음
        (6, 8, 10, 175.0, 15.0, 1.0),
        (5, 7,  9, 175.0, 15.0, 1.0),
        # 양팔이 일직선 — 오른손목-오른어깨-왼어깨
        (10, 6, 5, 175.0, 15.0, 1.2),
        (9,  5, 6, 175.0, 15.0, 1.2),
    ],
}


def calc_angle_score(action_name: str, kpts: np.ndarray) -> float:
    """각도 기준으로 점수 계산 (0~100). 기준 없으면 -1."""
    criteria = ACTION_ANGLE_CRITERIA.get(action_name)
    if criteria is None or kpts is None or kpts.shape[0] < 17:
        return -1.0

    scores, weights = [], []
    for (ai, bi, ci, target, tolerance, weight) in criteria:
        angle = calc_angle(kpts[ai], kpts[bi], kpts[ci])
        diff  = abs(angle - target)
        point = max(0.0, 1.0 - diff / tolerance) * 100.0
        scores.append(point)
        weights.append(weight)

    total_w = sum(weights)
    if total_w < 1e-6:
        return -1.0

    angle_sc = sum(s * w for s, w in zip(scores, weights)) / total_w

    # 위치 보조 점수와 7:3 합산
    pos_sc = calc_position_score(action_name, kpts)
    if pos_sc >= 0:
        return angle_sc * 0.7 + pos_sc * 0.3

    return angle_sc


# ================================================================
# 단일 단계 평가 모듈
# ================================================================

class StageEvaluator:
    """단일 동작 구간의 데이터를 수집하고 평가."""

    def __init__(
        self,
        stage_idx        : int, 
        action_name      : str,
        target_class_idx : int, # 몇번동작을 분석하는 놈인지
        detect_threshold : float = 0.3,
    ):
        self.stage_idx        = stage_idx
        self.action_name      = action_name
        self.target_class_idx = target_class_idx
        self.detect_threshold = detect_threshold

        self._scores_list : list = []
        self._kpts_list   : list = []
        self._collecting  = False
        self._frame_counter = 0

    # def start_collection(self):
    #     self._scores_list   = []
    #     self._kpts_list     = []
    #     self._frame_counter = 0
    #     self._collecting    = True
    #     print(f"[StageEval] 수집 시작 — stage {self.stage_idx}: "
    #           f"{self.action_name} (class_idx={self.target_class_idx})")

    def add_frame(
        self,
        scores_array: Optional[np.ndarray],
        keypoints   : Optional[np.ndarray] = None,
    ):
        """
        매 프레임 호출.
        scores_array : ndarray(num_classes,) 또는 None
        keypoints    : ndarray(133, 2) 또는 None
        """
        # if not self._collecting:
        #     return
        self._scores_list.append(
            scores_array if isinstance(scores_array, np.ndarray) else None
        )
        self._kpts_list.append(
            keypoints if isinstance(keypoints, np.ndarray) else None
        )
        self._frame_counter += 1

    # def is_collecting(self) -> bool:
    #     return self._collecting

    def analyze(self) -> EvalResult:
        """수집 종료 후 분석."""
        self._collecting = False
        idx          = self.target_class_idx
        total_frames = len(self._scores_list)

        if total_frames == 0:
            return EvalResult(
                stage_idx=self.stage_idx,
                action_name=self.action_name,
                target_class_idx=idx,
                conf_mean=0.0, conf_max=0.0,
                detect_ratio=0.0, first_detect_frame=-1,
                angle_score=-1.0, total_score=0.0,
                total_frames=0,
            )

        # ── conf 추출 ────────────────────────────────────────
        confs = []
        for s in self._scores_list:
            if s is not None and idx < len(s):
                confs.append(float(s[idx]))
            else:
                confs.append(0.0)

        confs_arr  = np.array(confs, dtype=float)
        conf_mean  = float(np.mean(confs_arr))
        conf_max   = float(np.max(confs_arr))

        detected_mask  = confs_arr >= self.detect_threshold
        detect_ratio   = float(np.sum(detected_mask)) / total_frames

        det_indices        = np.where(detected_mask)[0]
        first_detect_frame = int(det_indices[0]) if len(det_indices) > 0 else -1

        # # ── 각도 점수 ────────────────────────────────────────
        # # threshold 초과 프레임의 keypoints만 사용
        # angle_scores = []
        # for i, kpts in enumerate(self._kpts_list):
        #     if detected_mask[i] and kpts is not None:
        #         s = calc_angle_score(self.action_name, kpts)
        #         if s >= 0:
        #             angle_scores.append(s)

        # angle_score = float(np.mean(angle_scores)) if angle_scores else -1.0

        # ── 종합 점수 ────────────────────────────────────────
        angle_score = -1 #임시로
        total_score = self._calc_total(conf_mean, conf_max, detect_ratio, angle_score)

        result = EvalResult(
            stage_idx=self.stage_idx,
            action_name=self.action_name,
            target_class_idx=idx,
            conf_mean=round(conf_mean * 100, 1),
            conf_max=round(conf_max   * 100, 1),
            detect_ratio=round(detect_ratio * 100, 1),
            first_detect_frame=first_detect_frame,
            angle_score=round(angle_score, 1),
            total_score=round(total_score, 1),
            total_frames=total_frames,
        )
        print(f"[StageEval] 분석 완료: {result}")
        return result

    def _calc_total(
        self,
        conf_mean   : float,
        conf_max    : float,
        detect_ratio: float,
        angle_score : float,
    ) -> float:
        """
        가중치 기반 종합 점수 (0~100)
          keypoints 없음 → detect_ratio 35% | conf_mean 50% | conf_max 15%
          keypoints 있음 → detect_ratio 25% | conf_mean 45% | conf_max 10% | angle 20%
        """
        s_ratio = detect_ratio * 100
        s_mean  = conf_mean    * 100
        s_max   = conf_max     * 100

        if angle_score >= 0:
            score = s_ratio * 0.25 + s_mean * 0.45 + s_max * 0.10 + angle_score * 0.20
        else:
            score = s_ratio * 0.35 + s_mean * 0.50 + s_max * 0.15

        return score
        # return min(100.0, max(0.0, score))


# ================================================================
# 3단계 체험 전체 래퍼
# ================================================================

class ExperienceEvaluator:
    """
    3단계 체험 전체를 관리.
    StageEvaluator 3개를 묶어서 start/add/end 인터페이스 제공.
    """

    def __init__(
        self,
        stage_names      : dict, # list -> dict으로 수정
        stage_class_idxs : list,
        detect_threshold : float = 0.5,#0.3
    ):
        # assert len(stage_names) == len(stage_class_idxs), \
        #     "stage_names와 stage_class_idxs 길이가 같아야 합니다."
        self.stage_names      = stage_names
        self.stage_class_idxs = stage_class_idxs
        self.detect_threshold = detect_threshold

        self.stage_count      = len(stage_names) # 이것도 대체할게 있으면 뺄텐데... 아직 고민중
        self._current_stage     = -1 # 일단 나는 안씀
        
        self._evaluators : list = self._make_evaluators()
        self._results    : list = [None] * self.stage_count

    def _make_evaluators(self) -> list:
        return [
            StageEvaluator(
                stage_idx=i,
                action_name=self.stage_names[self.stage_class_idxs[i]],
                target_class_idx=self.stage_class_idxs[i],
                # target_class_idx=self.stage_class_idxs[i],
                detect_threshold=self.detect_threshold,
            )
            for i in range(len(self.stage_class_idxs))
            # for i in range(len(self.stage_names))
        ]

    # ── 수집 제어 ────────────────────────────────────────────────

    # def start_stage(self, stage_idx: int):
    #     if not (0 <= stage_idx < self.stage_count):
    #         print(f"[ExpEval] 잘못된 stage_idx: {stage_idx}")
    #         return
    #     self._current_stage = stage_idx
    #     self._evaluators[stage_idx].start_collection()

    def add_frame(
        self,
        action_flag : int,
        scores_array: Optional[np.ndarray],
        keypoints   : Optional[np.ndarray] = None,
    ):
        """
        매 프레임 호출.
        scores_array : action_results.get(active_user_id) — ndarray(num_classes,) or None
        keypoints    : keypoints_dict.get(active_user_id) — ndarray(133,2) or None
        """
        # if self._current_stage < 0:
        #     return
        self._evaluators[action_flag].add_frame(scores_array, keypoints)

    def end_stage(self, action_flag : int) -> Optional[EvalResult]:
        """구간 종료 → 분석 → EvalResult 반환"""
        # if self._current_stage < 0:
        #     return None
        result = self._evaluators[action_flag].analyze()
        self._results[action_flag] = result
        # self._results[self._current_stage] = result
        # self._current_stage = -1
        return result

    # def is_collecting(self) -> bool:
    #     if self._current_stage < 0:
    #         return False
    #     return self._evaluators[self._current_stage].is_collecting()

    # ── 결과 조회 ────────────────────────────────────────────────

    def get_summary(self) -> list:
        return self._results

    def get_total_score(self) -> float:
        valid = [r.total_score for r in self._results if r is not None]
        return round(sum(valid) / len(valid), 1) if valid else 0.0

    # ── 초기화 ──────────────────────────────────────────────────

    def reset(self):
        self._evaluators    = self._make_evaluators()
        self._results       = [None] * self.stage_count
        self._current_stage = -1
        print("[ExpEval] 초기화 완료")
