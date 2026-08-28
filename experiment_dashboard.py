"""NyquistGuard-TSC local desktop experiment controller.

The GUI never starts an experiment by itself. Pilot and full additionally need
an in-window confirmation before the runner receives its manual-start token.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from tkinter import BOTH, END, VERTICAL, BooleanVar, StringVar, Tk, messagebox
from tkinter import scrolledtext, ttk


PROJECT_ROOT = Path(__file__).resolve().parent
RUNNER_PATH = PROJECT_ROOT / "run_experiments.py"
RUNNER_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
STATE_PATH = PROJECT_ROOT / "runs" / "dashboard_status.json"
DIAGNOSTIC_REPORT_PATH = PROJECT_ROOT / "reports" / "diagnostic_report.md"
MECHANISM_REPORT_PATH = PROJECT_ROOT / "reports" / "mechanism_probe_report.md"
V3_10_CONFIRMATION_REPORT_PATH = PROJECT_ROOT / "reports" / "v3_10_independent_confirmation_report.md"
V4_MULTI_SEED_REPORT_PATH = PROJECT_ROOT / "reports" / "v4_residual_gate_multiseed_report.md"
V4_NEW_CONFIRMATION_REPORT_PATH = PROJECT_ROOT / "reports" / "v4_new_dataset_confirmation_report.md"
V5_MICRO_REPORT_PATH = PROJECT_ROOT / "reports" / "v5_dual_path_micro_report.md"
V5_BENCHMARK_REPORT_PATH = PROJECT_ROOT / "reports" / "v5_four_dataset_benchmark_report.md"
V5_1_INDEPENDENT_REPORT_PATH = PROJECT_ROOT / "reports" / "v5_1_independent_confirmation_report.md"
V5_1_FULL_EXTENSION_REPORT_PATH = PROJECT_ROOT / "reports" / "v5_1_full_extension_report.md"
FULL_REPORT_PATH = PROJECT_ROOT / "reports" / "full_report.md"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

STAGE_TASKS = {
    "smoke": [
        "环境与设备自检",
        "模型单元与合成回归",
        "BasicMotions 真实数据读取",
        "防泄漏与抗混叠多率视图",
        "CPU/CUDA 单批训练",
        "Checkpoint / Resume 检查",
        "Smoke 汇总报告",
    ],
    "pilot": [
        "准备 4 个 pilot 数据集",
        "执行 84 个可恢复 runs",
        "聚合指标与人工 Go/No-Go 报告",
    ],
    "diagnosis": [
        "定位并冻结有效 Pilot 结果",
        "审计 84-run 产物与协议一致性",
        "审计数据切分与标签闭集",
        "分析逐采样率性能与最差 rate",
        "分析 prediction flip 稳定性",
        "诊断选择性头与置信度排序",
        "检查训练历史、checkpoint 与消融",
        "专项检查 MiniROCKET 满分现象",
        "生成诊断报告与 Go/No-Go 结论",
    ],
    "mechanism_probe": [
        "预检 Pilot、checkpoint、配置与设备",
        "BasicMotions：gate/CBE/选择性/指纹探针",
        "Epilepsy：gate/CBE/选择性/指纹探针",
        "PAMAP2：gate/CBE/选择性/指纹探针",
        "MHEALTH：gate/CBE/选择性/指纹探针",
        "汇总候选根因与证据强度",
        "生成机制探针报告",
    ],
    "v3_10_independent_confirmation": [
        "训练 4 个全新 seed 的匹配 v1 控制",
        "训练 4 个全新 seed 的 v3.10 候选",
        "完成 4 组配对 validation 评估",
        "汇总冻结确认门并生成报告",
    ],
    "v4_residual_gate_multiseed_stability": [
        "冻结协议与train/validation零test访问预检",
        "BasicMotions seed42：硬gate同预算控制",
        "BasicMotions seed42：V4.1残余gate",
        "PAMAP2 seed2026：硬gate同预算控制",
        "PAMAP2 seed2026：V4.1残余gate",
        "BasicMotions seed2026：硬gate同预算控制",
        "BasicMotions seed2026：V4.1残余gate",
        "PAMAP2 seed42：硬gate同预算控制",
        "PAMAP2 seed42：V4.1残余gate",
        "汇总四组配对稳定性门与报告",
    ],
    "v4_new_dataset_confirmation": [
        item
        for dataset_id in (
            "character_trajectories_uea", "motor_imagery_uea",
            "wisdm_activity_uci", "ptbxl_physionet",
        )
        for item in (
            f"Prepare frozen {dataset_id}",
            *(f"{dataset_id} seed{seed} {role}"
              for seed in (16180, 57721, 94613)
              for role in ("v3_10_hard_gate", "v4_1_residual_gate")),
        )
    ] + ["Aggregate four-dataset confirmation and write report"],
    "v5_dual_path_micro": [
        "协议、资源互斥与train/validation边界预检",
        "BasicMotions seed17：冻结V4.1控制",
        "BasicMotions seed17：V5双路径候选",
        "PAMAP2 seed17：冻结V4.1控制",
        "PAMAP2 seed17：V5双路径候选",
        "汇总V5 validation开发门并生成报告",
    ],
    "v5_four_dataset_benchmark": [
        "验证V4.1来源、互斥锁与retrospective边界",
        *(
            item
            for dataset_id in (
                "character_trajectories_uea", "motor_imagery_uea",
                "wisdm_activity_uci", "ptbxl_physionet",
            )
            for item in (
                f"Load frozen {dataset_id}",
                *(f"{dataset_id} seed{seed} v5_dual_path"
                  for seed in (16180, 57721, 94613)),
            )
        ),
        "Aggregate V5 vs V4.1 four-dataset benchmark",
    ],
    "v5_1_independent_confirmation": [
        "Validate frozen panel, sources, files, and zero-TEST development caches",
        *(
            f"Validation train {dataset_id} seed{seed} {role}"
            for dataset_id in (
                "self_regulation_scp1_uea", "hand_movement_direction_uea",
                "racket_sports_uea", "heartbeat_uea",
            )
            for seed in (314159, 271828, 161803)
            for role in ("v4_1_residual_gate", "v5_dual_path")
        ),
        "Freeze four dataset-level reliability modes from validation only",
        *(
            f"Unlock official TEST {dataset_id}"
            for dataset_id in (
                "self_regulation_scp1_uea", "hand_movement_direction_uea",
                "racket_sports_uea", "heartbeat_uea",
            )
        ),
        *(
            f"One-shot TEST {dataset_id} seed{seed} {role}"
            for dataset_id in (
                "self_regulation_scp1_uea", "hand_movement_direction_uea",
                "racket_sports_uea", "heartbeat_uea",
            )
            for seed in (314159, 271828, 161803)
            for role in ("v4_1_residual_gate", "v5_dual_path")
        ),
        "Aggregate four-dataset independent confirmation and write report",
    ],
    "v5_1_full_extension": [
        "Validate frozen V5.1 extension and reused 210-run Full source",
        *(
            f"Train/validate {dataset_id} seed{seed} V5.1"
            for dataset_id in (
                "basicmotions_uea", "epilepsy_uea", "pamap2_uci", "mhealth_uci",
                "hapt_uci", "daily_sports_uci", "hydraulic_uci",
                "sleep_edfx_physionet", "eegmmi_physionet",
                "mitbih_arrhythmia_physionet",
            )
            for seed in (17, 42, 2026)
        ),
        "Freeze ten reliability modes from validation only",
        *(
            f"Evaluate TEST {dataset_id} seed{seed} V5.1"
            for dataset_id in (
                "basicmotions_uea", "epilepsy_uea", "pamap2_uci", "mhealth_uci",
                "hapt_uci", "daily_sports_uci", "hydraulic_uci",
                "sleep_edfx_physionet", "eegmmi_physionet",
                "mitbih_arrhythmia_physionet",
            )
            for seed in (17, 42, 2026)
        ),
        "Aggregate V5.1 with reused Full baselines and write report",
    ],
    "full": [
        "准备/复用 10 个冻结数据集缓存",
        "执行 210 个可断点续跑的 Full runs",
        "汇总配对统计、效率指标与 Full 报告",
    ],
    "full_parallel": [
        "复用当前 Full 父目录与 46 个已完成 runs",
        "Worker 0：固定偶数索引分片",
        "Worker 1：固定奇数索引分片",
        "按数据集 barrier 同步并汇总原 210-run 矩阵",
    ],
}


class FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", ctypes.c_uint32), ("dwHighDateTime", ctypes.c_uint32)]


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def _filetime_value(value: FILETIME) -> int:
    return (value.dwHighDateTime << 32) | value.dwLowDateTime


class SystemMetrics:
    """Low-overhead Windows metrics with a cached, slower GPU query."""

    def __init__(self, gpu_interval_seconds: float = 5.0) -> None:
        self._last_cpu: tuple[int, int] | None = None
        self._gpu_interval = float(gpu_interval_seconds)
        self._gpu_sampled_at = -float("inf")
        self._gpu_cache: tuple[float | None, float | None, float | None, str] = (
            None,
            None,
            None,
            "GPU 等待首次读取",
        )

    def cpu_percent(self) -> float | None:
        if os.name != "nt":
            return None
        idle, kernel, user = FILETIME(), FILETIME(), FILETIME()
        if not ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        ):
            return None
        idle_now = _filetime_value(idle)
        total_now = _filetime_value(kernel) + _filetime_value(user)
        previous = self._last_cpu
        self._last_cpu = (idle_now, total_now)
        if previous is None:
            return None
        idle_delta = idle_now - previous[0]
        total_delta = total_now - previous[1]
        if total_delta <= 0:
            return 0.0
        return max(0.0, min(100.0, 100.0 * (total_delta - idle_delta) / total_delta))

    @staticmethod
    def ram() -> tuple[float | None, float | None, float | None]:
        if os.name != "nt":
            return None, None, None
        memory = MEMORYSTATUSEX()
        memory.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            return None, None, None
        total_gb = memory.ullTotalPhys / (1024**3)
        used_gb = (memory.ullTotalPhys - memory.ullAvailPhys) / (1024**3)
        return float(memory.dwMemoryLoad), used_gb, total_gb

    @staticmethod
    def _query_gpu() -> tuple[float | None, float | None, float | None, str]:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,name",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=4,
                creationflags=CREATE_NO_WINDOW,
                check=True,
            )
            fields = [part.strip() for part in result.stdout.splitlines()[0].split(",", 3)]
            return float(fields[0]), float(fields[1]) / 1024, float(fields[2]) / 1024, fields[3]
        except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
            return None, None, None, "GPU 不可用"

    def gpu(self) -> tuple[float | None, float | None, float | None, str]:
        now = time.monotonic()
        if now - self._gpu_sampled_at >= self._gpu_interval:
            self._gpu_cache = self._query_gpu()
            self._gpu_sampled_at = now
        return self._gpu_cache

    def sample(self) -> dict[str, object]:
        cpu = self.cpu_percent()
        ram_percent, ram_used, ram_total = self.ram()
        gpu_percent, gpu_used, gpu_total, gpu_name = self.gpu()
        return {
            "cpu_percent": cpu,
            "ram_percent": ram_percent,
            "ram_used_gb": ram_used,
            "ram_total_gb": ram_total,
            "gpu_percent": gpu_percent,
            "gpu_used_gb": gpu_used,
            "gpu_total_gb": gpu_total,
            "gpu_name": gpu_name,
        }


class ExperimentDashboard:
    BG = "#08111f"
    PANEL = "#101c2e"
    PANEL_ALT = "#14233a"
    TEXT = "#edf4ff"
    MUTED = "#91a4bd"
    ACCENT = "#31d0aa"
    BLUE = "#4f8cff"
    AMBER = "#f2b84b"

    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("NyquistGuard-TSC 实验控制台")
        self.root.geometry("1140x780")
        self.root.minsize(980, 680)
        self.root.configure(bg=self.BG)
        self.root.option_add("*Font", "{Segoe UI} 10")
        self.process: subprocess.Popen[str] | None = None
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.stop_metrics = threading.Event()
        self.last_state_mtime: float | None = None
        self.stage_var = StringVar(value="mechanism_probe")
        self.resume_var = BooleanVar(value=True)
        self.status_var = StringVar(value="空闲")
        self.current_task_var = StringVar(value="等待手动开始")
        self.progress_text_var = StringVar(value="0 / 0")
        self.eta_var = StringVar(value="ETA —")
        self.runner_var = StringVar()
        self.metric_widgets: dict[str, tuple[ttk.Label, ttk.Progressbar, ttk.Label]] = {}
        self._configure_styles()
        self._build_ui()
        self._refresh_runner_badge()
        self._load_default_tasks()
        self._refresh_command_preview()
        self._start_metrics_thread()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(150, self._tick)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        for name, background in (("Root.TFrame", self.BG), ("Panel.TFrame", self.PANEL), ("PanelAlt.TFrame", self.PANEL_ALT)):
            style.configure(name, background=background)
        style.configure("Title.TLabel", background=self.BG, foreground=self.TEXT, font=("Segoe UI Semibold", 22))
        style.configure("Subtitle.TLabel", background=self.BG, foreground=self.MUTED)
        style.configure("PanelTitle.TLabel", background=self.PANEL, foreground=self.TEXT, font=("Segoe UI Semibold", 12))
        style.configure("Body.TLabel", background=self.PANEL, foreground=self.TEXT)
        style.configure("Muted.TLabel", background=self.PANEL, foreground=self.MUTED)
        style.configure("Metric.TLabel", background=self.PANEL_ALT, foreground=self.TEXT, font=("Segoe UI Semibold", 22))
        style.configure("MetricName.TLabel", background=self.PANEL_ALT, foreground=self.MUTED)
        style.configure("MetricDetail.TLabel", background=self.PANEL_ALT, foreground=self.MUTED, font=("Segoe UI", 9))
        style.configure("Accent.TButton", font=("Segoe UI Semibold", 11), padding=(18, 10))
        style.configure("Secondary.TButton", padding=(12, 9))
        for name, color in (("Main", self.ACCENT), ("CPU", self.BLUE), ("RAM", self.ACCENT), ("GPU", self.AMBER)):
            style.configure(f"{name}.Horizontal.TProgressbar", troughcolor="#20314b", background=color)
        style.configure(
            "Dashboard.Treeview",
            background=self.PANEL,
            fieldbackground=self.PANEL,
            foreground=self.TEXT,
            rowheight=28,
            borderwidth=0,
        )
        style.map("Dashboard.Treeview", background=[("selected", "#24456d")])

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, style="Root.TFrame", padding=22)
        outer.pack(fill=BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)
        header = ttk.Frame(outer, style="Root.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="NyquistGuard-TSC", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="本地实验控制台 · 所有实验只由你手动开始", style="Subtitle.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(header, textvariable=self.runner_var, style="Subtitle.TLabel").grid(row=0, column=1, rowspan=2, sticky="e")

        control = ttk.Frame(outer, style="Panel.TFrame", padding=18)
        control.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        control.columnconfigure(5, weight=1)
        ttk.Label(control, text="实验阶段", style="PanelTitle.TLabel").grid(row=0, column=0, padx=(0, 12))
        self.stage_box = ttk.Combobox(
            control,
            textvariable=self.stage_var,
            values=("smoke", "pilot", "diagnosis", "mechanism_probe", "v3_10_independent_confirmation", "v4_residual_gate_multiseed_stability", "v4_new_dataset_confirmation", "v5_dual_path_micro", "v5_four_dataset_benchmark", "v5_1_independent_confirmation", "v5_1_full_extension", "full", "full_parallel"),
            width=32,
            state="readonly",
        )
        self.stage_box.grid(row=0, column=1)
        self.stage_box.bind("<<ComboboxSelected>>", self._on_stage_change)
        ttk.Checkbutton(control, text="断点续跑 --resume", variable=self.resume_var, command=self._refresh_command_preview).grid(row=0, column=2, padx=16)
        self.start_button = ttk.Button(control, text="▶ 手动开始", style="Accent.TButton", command=self._start_run)
        self.start_button.grid(row=0, column=3, padx=(4, 8))
        self.stop_button = ttk.Button(control, text="■ 停止", style="Secondary.TButton", command=self._stop_run, state="disabled")
        self.stop_button.grid(row=0, column=4)
        ttk.Button(control, text="打开 Runs", style="Secondary.TButton", command=self._open_runs).grid(row=0, column=6, padx=(10, 0))
        self.report_button = ttk.Button(
            control,
            text="打开当前报告",
            style="Secondary.TButton",
            command=self._open_stage_report,
        )
        self.report_button.grid(row=0, column=7, padx=(8, 0))
        self.command_label = ttk.Label(control, text="", style="Muted.TLabel")
        self.command_label.grid(row=1, column=0, columnspan=8, sticky="w", pady=(12, 0))

        metrics = ttk.Frame(outer, style="Root.TFrame")
        metrics.grid(row=2, column=0, sticky="ew", pady=(0, 14))
        for column in range(3):
            metrics.columnconfigure(column, weight=1, uniform="metrics")
        self._build_metric_card(metrics, 0, "cpu", "CPU", "CPU.Horizontal.TProgressbar")
        self._build_metric_card(metrics, 1, "ram", "RAM", "RAM.Horizontal.TProgressbar")
        self._build_metric_card(metrics, 2, "gpu", "GPU", "GPU.Horizontal.TProgressbar")

        body = ttk.Frame(outer, style="Root.TFrame")
        body.grid(row=3, column=0, sticky="nsew")
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)
        left = ttk.Frame(body, style="Panel.TFrame", padding=18)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(5, weight=1)
        ttk.Label(left, text="当前进度", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        status_row = ttk.Frame(left, style="Panel.TFrame")
        status_row.grid(row=1, column=0, sticky="ew", pady=(12, 7))
        status_row.columnconfigure(0, weight=1)
        ttk.Label(status_row, textvariable=self.current_task_var, style="Body.TLabel", font=("Segoe UI Semibold", 13)).grid(row=0, column=0, sticky="w")
        ttk.Label(status_row, textvariable=self.status_var, style="Muted.TLabel").grid(row=0, column=1, sticky="e")
        self.main_progress = ttk.Progressbar(left, maximum=100, value=0, style="Main.Horizontal.TProgressbar")
        self.main_progress.grid(row=2, column=0, sticky="ew", pady=(0, 7))
        meta = ttk.Frame(left, style="Panel.TFrame")
        meta.grid(row=3, column=0, sticky="ew")
        meta.columnconfigure(0, weight=1)
        ttk.Label(meta, textvariable=self.progress_text_var, style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(meta, textvariable=self.eta_var, style="Muted.TLabel").grid(row=0, column=1, sticky="e")
        ttk.Label(left, text="运行日志", style="PanelTitle.TLabel").grid(row=4, column=0, sticky="w", pady=(18, 8))
        self.log = scrolledtext.ScrolledText(left, wrap="word", height=12, bg="#07101d", fg="#cfe0f4", insertbackground=self.TEXT, relief="flat", font=("Cascadia Mono", 9), padx=10, pady=10)
        self.log.grid(row=5, column=0, sticky="nsew")
        self.log.configure(state="disabled")

        right = ttk.Frame(body, style="Panel.TFrame", padding=18)
        right.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        ttk.Label(right, text="任务清单 / 剩余任务", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 10))
        self.task_tree = ttk.Treeview(right, columns=("status",), show="tree headings", style="Dashboard.Treeview", selectmode="none")
        self.task_tree.heading("#0", text="任务")
        self.task_tree.heading("status", text="状态")
        self.task_tree.column("#0", width=310, anchor="w")
        self.task_tree.column("status", width=80, anchor="center")
        task_scroll = ttk.Scrollbar(right, orient=VERTICAL, command=self.task_tree.yview)
        self.task_tree.configure(yscrollcommand=task_scroll.set)
        self.task_tree.grid(row=1, column=0, sticky="nsew")
        task_scroll.grid(row=1, column=1, sticky="ns")
        self._append_log("控制台已启动。选择阶段不会启动实验，必须点击“手动开始”。")

    def _build_metric_card(self, parent: ttk.Frame, column: int, key: str, title: str, style: str) -> None:
        card = ttk.Frame(parent, style="PanelAlt.TFrame", padding=15)
        card.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 7, 0 if column == 2 else 7))
        card.columnconfigure(0, weight=1)
        ttk.Label(card, text=title, style="MetricName.TLabel").grid(row=0, column=0, sticky="w")
        value = ttk.Label(card, text="—", style="Metric.TLabel")
        value.grid(row=1, column=0, sticky="w", pady=(3, 6))
        progress = ttk.Progressbar(card, maximum=100, value=0, style=style)
        progress.grid(row=2, column=0, sticky="ew")
        detail = ttk.Label(card, text="正在读取…", style="MetricDetail.TLabel")
        detail.grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.metric_widgets[key] = value, progress, detail

    def _refresh_runner_badge(self) -> None:
        python_ok = RUNNER_PYTHON.exists() or Path(sys.executable).exists()
        self.runner_var.set("● Runner 已连接" if RUNNER_PATH.exists() and python_ok else "● Runner 不可用")

    def _refresh_command_preview(self) -> None:
        python_name = RUNNER_PYTHON.name if RUNNER_PYTHON.exists() else Path(sys.executable).name
        command = f"{python_name} run_experiments.py --stage {self.stage_var.get()}"
        if self.resume_var.get():
            command += " --resume"
        if self.stage_var.get() in {"pilot", "v3_10_independent_confirmation", "v4_residual_gate_multiseed_stability", "v4_new_dataset_confirmation", "v5_dual_path_micro", "v5_four_dataset_benchmark", "v5_1_independent_confirmation", "v5_1_full_extension", "full", "full_parallel"}:
            command += " + 手动确认"
        self.command_label.configure(text=f"准备执行（不会自动运行）：{command}")

        if self.stage_var.get() in {"diagnosis", "mechanism_probe"}:
            self.command_label.configure(text=str(self.command_label.cget("text")) + "; read-only, no training")

    def _on_stage_change(self, _event: object | None = None) -> None:
        self._load_default_tasks()
        self._refresh_command_preview()

    def _load_default_tasks(self) -> None:
        tasks = [{"name": task, "status": "pending"} for task in STAGE_TASKS[self.stage_var.get()]]
        self._render_tasks(tasks)
        self.progress_text_var.set(f"0 / {len(tasks)}")
        self.main_progress.configure(mode="determinate", value=0)
        self.current_task_var.set("等待手动开始")
        self.status_var.set("空闲")
        self.eta_var.set("ETA —")

    def _render_tasks(self, tasks: list[dict[str, object]]) -> None:
        symbols = {"completed": "✓ 完成", "running": "▶ 进行中", "failed": "! 失败", "pending": "○ 等待"}
        for item in self.task_tree.get_children():
            self.task_tree.delete(item)
        for task in tasks:
            status = str(task.get("status", "pending"))
            self.task_tree.insert("", END, text=str(task.get("name", "未命名任务")), values=(symbols.get(status, status),))

    def _start_run(self) -> None:
        if self.process and self.process.poll() is None:
            return
        if not RUNNER_PATH.exists():
            messagebox.showerror("Runner 不存在", str(RUNNER_PATH), parent=self.root)
            return
        stage = self.stage_var.get()
        if stage in {"pilot", "v3_10_independent_confirmation", "v4_residual_gate_multiseed_stability", "v4_new_dataset_confirmation", "v5_dual_path_micro", "v5_four_dataset_benchmark", "v5_1_independent_confirmation", "v5_1_full_extension", "full", "full_parallel"}:
            if stage == "full_parallel":
                warning = (
                    "即将以两个独立 worker 继续当前 FULL。\n\n"
                    "两个 worker 使用同一 RTX 3060，但拥有互不重叠的冻结任务索引，并在每个数据集结束后同步。"
                    "模型、seed、batch、rate 和已有 checkpoint 不变。并发墙钟不可用于论文效率比较。\n\n"
                    "确认现在开始双 worker Full 吗？"
                )
            elif stage == "full":
                warning = (
                    "即将手动启动 FULL：10 个数据集、7 个方法、3 个 seeds，共 210 runs。\n\n"
                    "这是大型实验，可能连续运行数天。首次运行还会生成 6 个大型数据缓存。"
                    "关闭窗口或点击停止会终止当前进程，已落盘缓存、完整 run 和 checkpoint 会保留。\n\n"
                    "确认现在开始 Full 吗？"
                )
            elif stage == "v4_residual_gate_multiseed_stability":
                warning = (
                    "即将手动启动 V4.1 多 seed 稳定性实验。\n\n"
                    "实验包含 BasicMotions/PAMAP2 × seeds42/2026 × 硬gate/V4.1，共8个可恢复训练任务，"
                    "预计约18–25分钟。只读取train/validation，不会读取现有test，也不会启动Pilot或Full。\n\n"
                    "确认现在开始吗？"
                )
            elif stage == "v4_new_dataset_confirmation":
                warning = (
                    "即将手动启动 V4.1 四个全新数据集的正式确认实验。\n\n"
                    "实验包含 4 数据集 × 3 新 seeds × 硬 gate/V4.1，共 24 个可恢复训练任务，可能运行数小时。"
                    "点击确认后才会首次解锁冻结测试集；测试集不参与 checkpoint、阈值或可靠性模式选择。\n\n"
                    "确认现在开始吗？"
                )
            elif stage == "v5_dual_path_micro":
                warning = (
                    "即将手动启动 V5 validation-only 小型配对实验。\n\n"
                    "共4个可恢复训练任务，预计不超过约10分钟；只读取BasicMotions/PAMAP2的train/validation。"
                    "如果当前有其他stage正在运行，runner会拒绝启动且不会覆盖其进度。\n\n"
                    "确认现在开始吗？"
                )
            elif stage == "v5_four_dataset_benchmark":
                warning = (
                    "即将手动启动 V5 四数据集大型benchmark。\n\n"
                    "它复用当前正式实验的12个V4.1控制，只训练12个V5；支持断点续跑。"
                    "这些test已经被V4访问，因此结果不能称作V5全新独立确认。"
                    "如果任何其他stage仍在运行，runner会拒绝启动。\n\n"
                    "确认现在开始吗？"
                )
            elif stage == "v5_1_independent_confirmation":
                warning = (
                    "即将手动启动 V5.1 四个从未触碰TEST的数据集独立确认。\n\n"
                    "实验先完成4数据集×3 seeds×V4.1/V5的24个validation训练，再从validation冻结全部可靠性模式，"
                    "之后才首次读取TEST并一次性评估。共55个可恢复任务，预计需要数小时。\n\n"
                    "确认四个数据集的TRAIN和TEST文件均已下载，并现在开始吗？"
                )
            elif stage == "v5_1_full_extension":
                warning = (
                    "即将手动启动 V5.1 十数据集 Full 扩展。\n\n"
                    "只新增训练 30 个 V5.1 runs；原 210 个 Full 基线只读复用，不会重复训练。"
                    "支持断点续跑，预计属于大型实验。\n\n"
                    "确认现在开始吗？"
                )
            else:
                warning = (
                    f"即将由你手动启动 {stage.upper()}。\n\n"
                    "这会使用本机计算资源；关闭窗口或点击停止会终止当前进程，已落盘 checkpoint 会保留。\n\n"
                    "确认现在开始吗？"
                )
            if not messagebox.askyesno("确认手动开始", warning, parent=self.root):
                self._append_log(f"已取消 {stage}，没有启动任何进程。")
                return
        runner_python = RUNNER_PYTHON if RUNNER_PYTHON.exists() else Path(sys.executable)
        command = [str(runner_python), str(RUNNER_PATH), "--stage", stage]
        if self.resume_var.get():
            command.append("--resume")
        if stage in {"pilot", "v3_10_independent_confirmation", "v4_residual_gate_multiseed_stability", "v4_new_dataset_confirmation", "v5_dual_path_micro", "v5_four_dataset_benchmark", "v5_1_independent_confirmation", "v5_1_full_extension", "full", "full_parallel"}:
            command.append("--confirm-manual-start")
        runner_environment = os.environ.copy()
        runner_environment["PYTHONUTF8"] = "1"
        try:
            self.process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=runner_environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=CREATE_NO_WINDOW,
            )
        except OSError as error:
            messagebox.showerror("无法启动", str(error), parent=self.root)
            return
        self.status_var.set("运行中")
        self.current_task_var.set("等待 Runner 汇报进度")
        self.stage_box.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.main_progress.configure(mode="indeterminate")
        self.main_progress.start(12)
        self._append_log("已手动启动：" + " ".join(command))
        threading.Thread(target=self._read_process_output, daemon=True).start()

    def _read_process_output(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        process = self.process
        for line in process.stdout:
            self.events.put(("log", line.rstrip()))
        self.events.put(("process_end", process.wait()))

    def _stop_run(self) -> None:
        if not self.process or self.process.poll() is not None:
            return
        if messagebox.askyesno("停止实验", "确定停止当前进程吗？已落盘结果和 checkpoint 不会删除。", parent=self.root):
            self._append_log("正在停止实验进程…")
            self._terminate_process_tree()

    def _terminate_process_tree(self) -> None:
        if not self.process or self.process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                capture_output=True,
                creationflags=CREATE_NO_WINDOW,
                check=False,
            )
        else:
            self.process.terminate()

    def _handle_process_end(self, return_code: int) -> None:
        self.main_progress.stop()
        self.main_progress.configure(mode="determinate")
        self.stage_box.configure(state="readonly")
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        if return_code == 0:
            self.status_var.set("已完成")
            self.main_progress.configure(value=100)
            if self.stage_var.get() == "mechanism_probe":
                self.current_task_var.set(
                    "\u673a\u5236\u63a2\u9488\u5b8c\u6210\uff1b\u70b9\u51fb\u201c\u6253\u5f00\u5f53\u524d\u62a5\u544a\u201d\u67e5\u770b\u7ed3\u8bba"
                )
                self._append_log(
                    "\u673a\u5236\u63a2\u9488\u6b63\u5e38\u7ed3\u675f\u3002\u672a\u6784\u9020\u4f18\u5316\u5668\u3001\u672a\u66f4\u65b0\u53c2\u6570\u3001\u672a\u542f\u52a8 Pilot \u6216 Full\u3002"
                )
            if self.stage_var.get() == "diagnosis":
                self.current_task_var.set(
                    "\u8bca\u65ad\u5b8c\u6210\uff1b\u70b9\u51fb\u201c\u6253\u5f00\u8bca\u65ad\u62a5\u544a\u201d\u67e5\u770b\u7ed3\u8bba"
                )
                self._append_log(
                    "\u8bca\u65ad\u6b63\u5e38\u7ed3\u675f\u3002\u6ca1\u6709\u8bad\u7ec3\u6a21\u578b\uff0c\u4e5f\u4e0d\u4f1a\u81ea\u52a8\u542f\u52a8 Pilot \u6216 Full\u3002"
                )
            self._append_log("实验进程正常结束。不会自动启动下一阶段。")
        else:
            self.status_var.set("已停止 / 失败")
            self._append_log(f"进程结束，返回码 {return_code}；请查看日志。")
        self.process = None

    def _start_metrics_thread(self) -> None:
        def worker() -> None:
            metrics = SystemMetrics(gpu_interval_seconds=5.0)
            while not self.stop_metrics.is_set():
                self.events.put(("metrics", metrics.sample()))
                self.stop_metrics.wait(2.0)

        threading.Thread(target=worker, daemon=True).start()

    def _update_metrics(self, data: dict[str, object]) -> None:
        self._set_metric("cpu", data.get("cpu_percent"), "系统总占用（每 2 秒）")
        ram_detail = "内存不可用"
        if data.get("ram_used_gb") is not None and data.get("ram_total_gb") is not None:
            ram_detail = f"{float(data['ram_used_gb']):.1f} / {float(data['ram_total_gb']):.1f} GB"
        self._set_metric("ram", data.get("ram_percent"), ram_detail)
        gpu_detail = str(data.get("gpu_name", "GPU 不可用"))
        if data.get("gpu_used_gb") is not None and data.get("gpu_total_gb") is not None:
            gpu_detail += f" · VRAM {float(data['gpu_used_gb']):.1f} / {float(data['gpu_total_gb']):.1f} GB · 每 5 秒"
        self._set_metric("gpu", data.get("gpu_percent"), gpu_detail)

    def _set_metric(self, key: str, percent: object, detail: str) -> None:
        value_label, progress, detail_label = self.metric_widgets[key]
        if percent is None:
            value_label.configure(text="—")
            progress.configure(value=0)
        else:
            numeric = max(0.0, min(100.0, float(percent)))
            value_label.configure(text=f"{numeric:.0f}%")
            progress.configure(value=numeric)
        detail_label.configure(text=detail)

    def _poll_state_file(self) -> None:
        if not STATE_PATH.exists():
            return
        try:
            mtime = STATE_PATH.stat().st_mtime
            if mtime == self.last_state_mtime:
                return
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            self.last_state_mtime = mtime
        except (OSError, json.JSONDecodeError):
            return
        if state.get("stage") != self.stage_var.get():
            return
        self.status_var.set(str(state.get("status", self.status_var.get())))
        self.current_task_var.set(str(state.get("current_task", self.current_task_var.get())))
        completed = int(state.get("completed_tasks", 0))
        total = int(state.get("total_tasks", 0))
        progress = float(state.get("progress_percent", 100 * completed / total if total else 0))
        self.main_progress.stop()
        self.main_progress.configure(mode="determinate", value=max(0, min(100, progress)))
        self.progress_text_var.set(f"{completed} / {total} · {progress:.1f}%")
        eta = state.get("eta_seconds")
        self.eta_var.set("ETA —" if eta is None else f"ETA {self._format_duration(float(eta))}")
        tasks = state.get("tasks")
        if isinstance(tasks, list):
            self._render_tasks([item for item in tasks if isinstance(item, dict)])

    @staticmethod
    def _format_duration(seconds: float) -> str:
        hours, remainder = divmod(max(0, int(seconds)), 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {secs:02d}s"

    def _append_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert(END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.log.see(END)
        self.log.configure(state="disabled")

    def _open_runs(self) -> None:
        runs = PROJECT_ROOT / "runs"
        runs.mkdir(exist_ok=True)
        if os.name == "nt":
            os.startfile(runs)  # type: ignore[attr-defined]

    def _open_stage_report(self) -> None:
        if self.stage_var.get() in {"full", "full_parallel"}:
            if not FULL_REPORT_PATH.exists():
                messagebox.showinfo(
                    "尚无 Full 报告",
                    "Full 必须由你单独手动确认并启动；运行支持断点续跑。",
                    parent=self.root,
                )
                return
            if os.name == "nt":
                os.startfile(FULL_REPORT_PATH)  # type: ignore[attr-defined]
            return
        if self.stage_var.get() == "v3_10_independent_confirmation":
            if not V3_10_CONFIRMATION_REPORT_PATH.exists():
                messagebox.showinfo(
                    "尚无 v3.10 独立确认报告",
                    "请先手动启动 v3_10_independent_confirmation；实验支持断点续跑。",
                    parent=self.root,
                )
                return
            if os.name == "nt":
                os.startfile(V3_10_CONFIRMATION_REPORT_PATH)  # type: ignore[attr-defined]
            return
        if self.stage_var.get() == "v4_residual_gate_multiseed_stability":
            if not V4_MULTI_SEED_REPORT_PATH.exists():
                messagebox.showinfo(
                    "尚无 V4.1 多 seed 报告",
                    "请先手动启动 v4_residual_gate_multiseed_stability；实验支持断点续跑。",
                    parent=self.root,
                )
                return
            if os.name == "nt":
                os.startfile(V4_MULTI_SEED_REPORT_PATH)  # type: ignore[attr-defined]
            return
        if self.stage_var.get() == "v4_new_dataset_confirmation":
            if not V4_NEW_CONFIRMATION_REPORT_PATH.exists():
                messagebox.showinfo(
                    "尚无 V4.1 新数据集确认报告",
                    "请先手动启动 v4_new_dataset_confirmation；实验支持断点续跑。",
                    parent=self.root,
                )
                return
            if os.name == "nt":
                os.startfile(V4_NEW_CONFIRMATION_REPORT_PATH)  # type: ignore[attr-defined]
            return
        if self.stage_var.get() == "v5_dual_path_micro":
            if not V5_MICRO_REPORT_PATH.exists():
                messagebox.showinfo(
                    "尚无 V5 小型实验报告",
                    "当前V4完成后再手动启动 v5_dual_path_micro；支持断点续跑。",
                    parent=self.root,
                )
                return
            if os.name == "nt":
                os.startfile(V5_MICRO_REPORT_PATH)  # type: ignore[attr-defined]
            return
        if self.stage_var.get() == "v5_four_dataset_benchmark":
            if not V5_BENCHMARK_REPORT_PATH.exists():
                messagebox.showinfo(
                    "尚无 V5 四数据集benchmark报告",
                    "必须先完成当前V4，再手动启动 v5_four_dataset_benchmark。",
                    parent=self.root,
                )
                return
            if os.name == "nt":
                os.startfile(V5_BENCHMARK_REPORT_PATH)  # type: ignore[attr-defined]
            return
        if self.stage_var.get() == "v5_1_independent_confirmation":
            if not V5_1_INDEPENDENT_REPORT_PATH.exists():
                messagebox.showinfo(
                    "尚无 V5.1 独立确认报告",
                    "请先下载冻结面板数据，再手动启动 v5_1_independent_confirmation；支持断点续跑。",
                    parent=self.root,
                )
                return
            if os.name == "nt":
                os.startfile(V5_1_INDEPENDENT_REPORT_PATH)  # type: ignore[attr-defined]
            return
        if self.stage_var.get() == "v5_1_full_extension":
            if not V5_1_FULL_EXTENSION_REPORT_PATH.exists():
                messagebox.showinfo(
                    "尚无 V5.1 Full 扩展报告",
                    "请先手动启动 v5_1_full_extension；支持断点续跑。",
                    parent=self.root,
                )
                return
            if os.name == "nt":
                os.startfile(V5_1_FULL_EXTENSION_REPORT_PATH)  # type: ignore[attr-defined]
            return
        if self.stage_var.get() == "mechanism_probe":
            if not MECHANISM_REPORT_PATH.exists():
                messagebox.showinfo(
                    "\u5c1a\u65e0\u673a\u5236\u63a2\u9488\u62a5\u544a",
                    "\u8bf7\u5148\u9009\u62e9 mechanism_probe \u5e76\u70b9\u51fb\u201c\u624b\u52a8\u5f00\u59cb\u201d\u3002",
                    parent=self.root,
                )
                return
            if os.name == "nt":
                os.startfile(MECHANISM_REPORT_PATH)  # type: ignore[attr-defined]
            return
        if not DIAGNOSTIC_REPORT_PATH.exists():
            messagebox.showinfo(
                "\u5c1a\u65e0\u8bca\u65ad\u62a5\u544a",
                "\u8bf7\u5148\u9009\u62e9 diagnosis \u5e76\u70b9\u51fb\u201c\u624b\u52a8\u5f00\u59cb\u201d\u3002\u8bca\u65ad\u4e0d\u4f1a\u8bad\u7ec3\u6a21\u578b\u3002",
                parent=self.root,
            )
            return
        if os.name == "nt":
            os.startfile(DIAGNOSTIC_REPORT_PATH)  # type: ignore[attr-defined]

    def _tick(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "metrics":
                    self._update_metrics(payload)  # type: ignore[arg-type]
                elif event == "log":
                    self._append_log(str(payload))
                elif event == "process_end":
                    self._handle_process_end(int(payload))
        except queue.Empty:
            pass
        self._poll_state_file()
        self.root.after(250, self._tick)

    def _on_close(self) -> None:
        if self.process and self.process.poll() is None:
            if not messagebox.askyesno("实验仍在运行", "关闭窗口会停止当前进程。确定关闭吗？", parent=self.root):
                return
            self._terminate_process_tree()
        self.stop_metrics.set()
        self.root.destroy()


def self_test() -> int:
    monitor = SystemMetrics(gpu_interval_seconds=5.0)
    monitor.cpu_percent()
    time.sleep(0.15)
    data = monitor.sample()
    required = {"cpu_percent", "ram_percent", "gpu_percent", "gpu_name"}
    if set(data) < required:
        raise RuntimeError("metrics self-test returned an incomplete payload")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"runner_exists={RUNNER_PATH.exists()}")
    print(f"runner_python={RUNNER_PYTHON if RUNNER_PYTHON.exists() else Path(sys.executable)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="NyquistGuard-TSC local dashboard")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    root = Tk()
    ExperimentDashboard(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
