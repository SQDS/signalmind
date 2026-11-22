import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Literal, Optional, Tuple
import numpy as np
from scipy.ndimage import uniform_filter1d, binary_closing


@dataclass
class ArtifactEvent:
    id: int
    type: Literal[
        "EMG_OR_MOTION_ARTIFACT",
        "BASELINE_DRIFT",
        "LEAD_OFF_OR_SATURATION",
        "EXTREME_AMPLITUDE_SPIKE",
    ]
    time_window_sec: Tuple[float, float]
    sample_indices: Tuple[int, int]
    severity_score: float
    dominant_metric: str
    geometric_cause: str


class TimeSphereArtifactDetector:
    """
    3D 时间球生理信号噪声与伪影定位引擎（Detector-Free Topological SQI）
    特性：
      1. 零篡改：只读分析，不修改、不平滑原始 1D 信号。
      2. 免 R 波检测：采用短时滑动自相关（ST-ACF）动态估计瞬时主周期，抗变心率漂移。
      3. 四维几何算子：在 K x M 球面规则网格上并行提取跨圈粗糙度、扇区体积偏移、死线平坦度与外包络尖峰。
      4. 双向索引逆映射：自动输出精确时间窗 [t_start, t_end] 及供前端渲染的 3D 网格热力数据。
    """

    def __init__(
        self,
        fs: float = 125.0,
        r_base: float = 1.0,
        radial_scale: float = 0.25,
        phase_bins: int = 128,
        hr_range_bpm: Tuple[float, float] = (42.0, 165.0),
        sensitivity: Literal["strict", "balanced", "loose"] = "balanced",
    ):
        self.fs = float(fs)
        self.r_base = float(r_base)
        self.radial_scale = float(radial_scale)
        self.phase_bins = int(phase_bins)  # 经度网格分辨率 M
        self.hr_range_bpm = hr_range_bpm
        self.sensitivity = sensitivity

        self._thresh_map = {
            "strict": 2.4,
            "balanced": 3.2,
            "loose": 4.2,
        }

    @staticmethod
    def _mad_normalize(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """基于中位数绝对偏差（MAD）的鲁棒无参数标准化"""
        med = np.median(arr)
        mad = np.median(np.abs(arr - med)) + eps
        return np.maximum(0.0, (arr - med) / (1.4826 * mad))

    def _estimate_adaptive_phase(self, z_sig: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        短时滑动窗口自相关（ST-ACF）估计瞬时主周期，生成连续单调递增的累积相位 phi_cum (单位: 圈数)
        """
        n = len(z_sig)
        min_lag = max(2, int(self.fs * 60.0 / self.hr_range_bpm[1]))
        max_lag = min(n // 2, int(self.fs * 60.0 / self.hr_range_bpm[0]))

        def _acf_period(seg: np.ndarray) -> float:
            seg = seg - np.mean(seg)
            if np.std(seg) < 1e-6 or len(seg) <= max_lag:
                return self.fs * 0.8
            acf = np.correlate(seg, seg, mode="full")[len(seg) - 1 :]
            acf = acf / (acf[0] + 1e-8)
            search_win = acf[min_lag:max_lag]
            if len(search_win) == 0:
                return self.fs * 0.8
            best_lag = min_lag + int(np.argmax(search_win))
            return float(best_lag)

        global_t0 = _acf_period(z_sig)

        # 若信号足够长（> 8秒），每 4 秒窗口追踪一次瞬时主周期并平滑插值，适应运动/变心率
        win_samples = int(self.fs * 6.0)
        hop_samples = int(self.fs * 3.0)
        if n > win_samples * 1.5:
            centers, periods = [], []
            for start in range(0, n - win_samples + 1, hop_samples):
                seg = z_sig[start : start + win_samples]
                p = _acf_period(seg)
                # 防止局部严重脏数据导致周期倍频跳变，约束在全局主周期的 [0.65, 1.45] 倍内
                if 0.65 * global_t0 <= p <= 1.45 * global_t0:
                    centers.append(start + win_samples / 2.0)
                    periods.append(p)
            if len(centers) >= 2:
                inst_period = np.interp(np.arange(n), centers, periods)
                inst_period = uniform_filter1d(inst_period, size=int(self.fs * 2.0))
            else:
                inst_period = np.full(n, global_t0)
        else:
            inst_period = np.full(n, global_t0)

        # 累积圈数相位（连续单调递增）
        d_cycle = 1.0 / np.maximum(inst_period, 1.0)
        cum_cycles = np.cumsum(d_cycle)
        cum_cycles -= cum_cycles[0]
        return cum_cycles, global_t0

    def detect(
        self,
        raw_signal: np.ndarray,
        include_mesh: bool = True,
    ) -> Dict:
        """
        执行完整时间球构球、四维几何异常打分及 1D 区间逆映射
        """
        sig = np.asarray(raw_signal, dtype=np.float64).flatten()
        n = len(sig)
        if n < int(self.fs * 3):
            raise ValueError("信号长度过短（建议至少 3 秒以上以形成多圈球面拓扑）")

        # 1. 鲁棒幅度标准化（使用中位数与 IQR，防止极端伪影拉偏基线）
        q25, q50, q75 = np.percentile(sig, [25, 50, 75])
        iqr = (q75 - q25) if (q75 - q25) > 1e-8 else (np.std(sig) + 1e-8)
        z_sig = (sig - q50) / iqr

        # 2. 自适应累积相位与 K x M 规则球面网格构建
        cum_cycles, global_t0 = self._estimate_adaptive_phase(z_sig)
        k_cycles = max(4, int(np.floor(cum_cycles[-1])))
        m_bins = self.phase_bins

        # 规则网格目标相位坐标: (k, m) -> target_cycle = k + m / M
        grid_k, grid_m = np.meshgrid(
            np.arange(k_cycles), np.arange(m_bins), indexing="ij"
        )
        target_cycles = (grid_k + grid_m / float(m_bins)).ravel()

        # 插值得到每个球面网格顶点的径向幅值及对应的原始采样点索引
        grid_z = np.interp(target_cycles, cum_cycles, z_sig).reshape(k_cycles, m_bins)
        grid_sample_idx = np.clip(
            np.round(np.interp(target_cycles, cum_cycles, np.arange(n))).astype(int),
            0,
            n - 1,
        ).reshape(k_cycles, m_bins)

        r_grid = self.r_base + self.radial_scale * grid_z

        # 3. 四维正交几何异常算子计算（核心算法）
        # (A) 跨圈各向异性粗糙度 S_cross：沿经线（跨纬度 k 方向）计算二阶差分 + 经向中位数偏离
        #     正常 R 波在上下圈重复出现，差值极小；肌电/运动伪影在上下圈乱跳，数值飙升
        d2_lat = np.zeros_like(r_grid)
        d2_lat[1:-1, :] = np.abs(r_grid[2:, :] - 2.0 * r_grid[1:-1, :] + r_grid[:-2, :])
        d2_lat[0, :] = np.abs(r_grid[1, :] - r_grid[0, :])
        d2_lat[-1, :] = np.abs(r_grid[-1, :] - r_grid[-2, :])

        meridian_median = np.median(r_grid, axis=0, keepdims=True)
        meridian_dev = np.abs(r_grid - meridian_median)
        s_cross = 0.6 * d2_lat + 0.4 * meridian_dev

        # (B) 局部扇区体积偏移 dV_local：3圈 x 16相位窗内的径向三次方偏移 |mean(r^3) - R0^3|
        #     专抓呼吸/体位导致的低频基线漂移（表面光滑但局部体积鼓包或塌陷）
        r_cubed_diff = r_grid**3 - self.r_base**3
        # 沿经度做循环卷积均值，沿纬度做滑窗均值
        pad_m = 8
        padded_v = np.pad(r_cubed_diff, ((1, 1), (pad_m, pad_m)), mode="wrap")
        dv_local = np.zeros_like(r_grid)
        for dk in range(3):
            for dm in range(2 * pad_m + 1):
                dv_local += padded_v[dk : dk + k_cycles, dm : dm + m_bins]
        dv_local = np.abs(dv_local / (3.0 * (2 * pad_m + 1)))

        # (C) 局部零起伏与山脊断裂 F_flat：专抓导联脱落死线或硬件削波饱和
        #     局部同圈一阶差分方差趋近 0，同时该圈峰值显著低于正常参考山脊高度
        d1_lon = np.abs(np.diff(r_grid, axis=1, prepend=r_grid[:, -1:]))
        local_rough_lon = uniform_filter1d(d1_lon, size=max(4, m_bins // 8), axis=1, mode="wrap")
        ref_ridge_height = float(np.percentile(meridian_median - self.r_base, 92))
        cycle_peak_height = np.max(r_grid - self.r_base, axis=1, keepdims=True)
        ridge_loss = np.maximum(0.0, 1.0 - cycle_peak_height / (ref_ridge_height + 1e-6))
        f_flat = np.where(local_rough_lon < 0.004 * self.radial_scale, 1.0 + 3.0 * ridge_loss, 0.0)

        # (D) 径向极端外包络越界 F_spike：兜底抓电极瞬间松动引起的超大跳变
        f_spike = np.maximum(0.0, np.abs(grid_z) - 4.5)

        # 4. 鲁棒标准化与综合异常热力评分融合
        score_rough = self._mad_normalize(s_cross)
        score_vol = self._mad_normalize(dv_local)
        score_flat = f_flat * 4.0
        score_spike = f_spike * 2.5

        total_score_grid = np.maximum.reduce(
            [score_rough, 0.85 * score_vol, score_flat, score_spike]
        )

        # 5. 网格异常点反向投影至 1D 采样点时间轴
        thresh = self._thresh_map.get(self.sensitivity, 3.2)
        anomaly_grid_mask = total_score_grid > thresh

        sample_score = np.zeros(n, dtype=np.float64)
        sample_type_code = np.zeros(n, dtype=np.int32)  # 1:EMG/Motion, 2:Drift, 3:Dead, 4:Spike

        # 确定每个网格点的主导异常类型
        stack_scores = np.stack(
            [score_rough, 0.85 * score_vol, score_flat, score_spike], axis=0
        )
        dominant_type_grid = np.argmax(stack_scores, axis=0) + 1

        flat_indices = grid_sample_idx.ravel()
        flat_scores = total_score_grid.ravel()
        flat_types = dominant_type_grid.ravel()
        flat_mask = anomaly_grid_mask.ravel()

        for idx, sc, tp, is_anom in zip(flat_indices, flat_scores, flat_types, flat_mask):
            if sc > sample_score[idx]:
                sample_score[idx] = sc
                if is_anom:
                    sample_type_code[idx] = tp

        # 将稀疏网格投影填充并做形态学闭运算（合并间隔小于 0.35s 的碎片警报）
        raw_mask_1d = sample_score > thresh
        gap_samples = max(1, int(self.fs * 0.35))
        closed_mask_1d = binary_closing(raw_mask_1d, structure=np.ones(gap_samples))

        # 6. 提取连续异常区间 [t_start, t_end]
        events: List[ArtifactEvent] = []
        padded_mask = np.concatenate([[False], closed_mask_1d, [False]])
        starts = np.where(padded_mask[1:] & ~padded_mask[:-1])[0]
        ends = np.where(~padded_mask[1:] & padded_mask[:-1])[0]

        type_meta = {
            1: (
                "EMG_OR_MOTION_ARTIFACT",
                "Cross-Cycle Surface Roughness (S_cross)",
                "跨圈经向二阶差分剧增，相邻周期相位一致性破坏",
            ),
            2: (
                "BASELINE_DRIFT",
                "Local Sector Volume Shift (ΔV_local)",
                "局部球面扇区径向体积偏移超限，曲面光滑但整体鼓包/塌陷",
            ),
            3: (
                "LEAD_OFF_OR_SATURATION",
                "Zero-Roughness & Ridge Collapse (F_flat)",
                "局部曲面粗糙度归零且纵贯 R 波山脊断裂",
            ),
            4: (
                "EXTREME_AMPLITUDE_SPIKE",
                "Radial Envelope Exceedance (F_spike)",
                "径向幅值突破极端外包络阈值",
            ),
        }

        for ev_id, (s_idx, e_idx) in enumerate(zip(starts, ends), start=1):
            e_idx = min(n - 1, e_idx)
            if e_idx - s_idx < max(2, int(self.fs * 0.08)):
                continue
            seg_types = sample_type_code[s_idx : e_idx + 1]
            valid_types = seg_types[seg_types > 0]
            dom_code = int(np.bincount(valid_types).argmax()) if len(valid_types) > 0 else 1
            ev_type, dom_metric, cause = type_meta[dom_code]
            max_sc = float(np.max(sample_score[s_idx : e_idx + 1]))
            severity = round(min(1.0, max_sc / (thresh * 2.2)), 3)

            events.append(
                ArtifactEvent(
                    id=ev_id,
                    type=ev_type,
                    time_window_sec=(round(s_idx / self.fs, 3), round(e_idx / self.fs, 3)),
                    sample_indices=(int(s_idx), int(e_idx)),
                    severity_score=severity,
                    dominant_metric=dom_metric,
                    geometric_cause=cause,
                )
            )

        clean_ratio = float(1.0 - np.mean(closed_mask_1d))
        # 综合信号质量指数 SQI
        overall_sqi = round(max(0.0, min(1.0, clean_ratio * (1.0 - 0.15 * len(events) / max(1, k_cycles)))), 3)

        result = {
            "status": "SUCCESS",
            "sampling_rate_hz": self.fs,
            "duration_sec": round(n / self.fs, 3),
            "estimated_period_sec": round(global_t0 / self.fs, 4),
            "estimated_hr_bpm": round(60.0 * self.fs / global_t0, 1),
            "sphere_topology": {"latitude_cycles_K": k_cycles, "longitude_bins_M": m_bins},
            "overall_sqi": overall_sqi,
            "clean_ratio": round(clean_ratio, 4),
            "artifact_count": len(events),
            "recommendation": (
                "PASS_DIRECTLY"
                if overall_sqi >= 0.90
                else ("EXCLUDE_ARTIFACT_WINDOWS" if overall_sqi >= 0.50 else "REJECT_SEGMENT_POOR_QUALITY")
            ),
            "detected_artifacts": [asdict(ev) for ev in events],
        }

        if include_mesh:
            # 导出供前端 3D 雷达直接渲染的球面网格坐标、热力值与时间索引
            theta = np.linspace(0.12, np.pi - 0.12, k_cycles)[:, None]
            phi = np.linspace(0.0, 2.0 * np.pi, m_bins, endpoint=False)[None, :]
            x = r_grid * np.sin(theta) * np.cos(phi)
            y = r_grid * np.cos(theta)
            z = r_grid * np.sin(theta) * np.sin(phi)

            result["visualization_payload"] = {
                "r_base": self.r_base,
                "raw_signal": np.round(sig, 4).tolist(),
                "sample_mask_1d": closed_mask_1d.astype(int).tolist(),
                "mesh_x": np.round(x, 4).tolist(),
                "mesh_y": np.round(y, 4).tolist(),
                "mesh_z": np.round(z, 4).tolist(),
                "mesh_anomaly_score": np.round(total_score_grid / thresh, 3).tolist(),
                "mesh_dominant_type": dominant_type_grid.tolist(),
                "mesh_sample_idx": grid_sample_idx.tolist(),
            }

        return result


# ==========================================
# AI Agent Tool 调用接口示例 (可直接注册给 LLM)
# ==========================================
def agent_tool_locate_signal_artifacts(
    signal_list: List[float],
    sampling_rate: float = 125.0,
    sensitivity: Literal["strict", "balanced", "loose"] = "balanced",
) -> str:
    """
    [Agent Tool] 使用 3D 时间球拓扑几何分析定位生理信号中的所有噪声与伪影时间段。
    返回结构化 JSON 诊断报告（不含冗余网格坐标以节省 LLM Token）。
    """
    detector = TimeSphereArtifactDetector(fs=sampling_rate, sensitivity=sensitivity)
    report = detector.detect(np.array(signal_list), include_mesh=False)
    return json.dumps(report, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 本地自测：生成 20 秒 125Hz 模拟 ECG 并注入三类噪声，验证盲检输出
    fs = 125.0
    t = np.arange(0, 20.0, 1.0 / fs)
    hr_hz = 1.25  # 75 bpm
    phase = (t * hr_hz) % 1.0
    ecg = (
        0.12 * np.exp(-((phase - 0.16) / 0.025) ** 2)
        - 0.15 * np.exp(-((phase - 0.28) / 0.012) ** 2)
        + 1.15 * np.exp(-((phase - 0.31) / 0.014) ** 2)
        - 0.25 * np.exp(-((phase - 0.34) / 0.015) ** 2)
        + 0.28 * np.exp(-((phase - 0.56) / 0.045) ** 2)
        + 0.015 * np.random.randn(len(t))
    )

    # 注入未知干扰：1) 高频肌电 (4.5-6.2s)  2) 基线漂移 (11.0-14.0s)  3) 导联死线 (17.0-18.5s)
    ecg[(t >= 4.5) & (t <= 6.2)] += 0.45 * np.random.randn(np.sum((t >= 4.5) & (t <= 6.2)))
    ecg[(t >= 11.0) & (t <= 14.0)] += 0.85 * np.sin(2 * np.pi * 0.35 * t[(t >= 11.0) & (t <= 14.0)])
    ecg[(t >= 17.0) & (t <= 18.5)] = 0.0

    detector = TimeSphereArtifactDetector(fs=fs, sensitivity="balanced")
    res = detector.detect(ecg, include_mesh=True)

    # 打印供 Agent 消费的纯文本摘要
    summary = {k: v for k, v in res.items() if k != "visualization_payload"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 导出前端可视化数据包供 HTML 加载
    with open("timesphere_result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    print("\n已导出 timesphere_result.json，可直接加载至前端 3D 雷达界面。")
