# -*- coding: utf-8 -*-
"""
图片文字识别 (OCR) 图形界面
=================================================

基于 ocr_cli.py 的 Tkinter 图形界面，功能与命令行版一致：

  - 选择图片并预览
  - 可选填写标注文本（ground truth），自动计算识别准确率
  - 展示每行识别结果与置信度，低置信度行高亮
  - 展示字符/词/行级准确率与「标注 vs 识别」差异对照
  - 可导出为 TXT 或 JSON

启动:
  python ocr_gui.py

依赖:
  ocr_cli.py（核心逻辑）、rapidocr-onnxruntime、pillow
"""

from __future__ import annotations

import importlib.util
import json
import os
import queue
import sys
import threading
import traceback
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
# 加载同目录下的核心模块 ocr_cli.py
# ---------------------------------------------------------------------------


def _resource_dir() -> str:
    """资源目录：打包成 exe 后文件被解压到 sys._MEIPASS，开发时用脚本所在目录。"""
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


_CORE_PATH = os.path.join(_resource_dir(), "ocr_cli.py")


def load_core():
    """按文件路径加载核心模块。

    不用普通的 `import ocr_cli`：打包成 exe 后，模块并不以可导入包的形式
    存在，而是被解压到临时目录，按路径加载才能同时兼顾开发运行与打包运行。
    """
    if not os.path.exists(_CORE_PATH):
        raise FileNotFoundError(f"找不到核心模块: {_CORE_PATH}")
    spec = importlib.util.spec_from_file_location("ocr_core", _CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ocr_core"] = module
    spec.loader.exec_module(module)
    return module


try:
    core = load_core()
except Exception as exc:  # pragma: no cover
    tk.Tk().withdraw()
    messagebox.showerror("启动失败", f"无法加载核心模块:\n{exc}")
    raise SystemExit(1)


IMAGE_TYPES = [
    ("图片文件", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),
    ("所有文件", "*.*"),
]

ENGINE_CHOICES = [
    ("自动选择", "auto"),
    ("RapidOCR", "rapidocr"),
    ("Tesseract", "pytesseract"),
]


def setup_dpi_awareness() -> None:
    """让界面在高 DPI 屏幕上不发虚。"""
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def pick_ui_font(root: tk.Misc) -> str:
    """优先使用支持中文的字体。"""
    from tkinter import font as tkfont

    try:
        families = set(tkfont.families(root))
    except Exception:
        return "TkDefaultFont"
    for name in ("Microsoft YaHei UI", "Microsoft YaHei", "微软雅黑", "SimHei"):
        if name in families:
            return name
    return "TkDefaultFont"


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------


class OcrApp(ttk.Frame):
    PREVIEW_W, PREVIEW_H = 380, 420

    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master, padding=8)
        self.master: tk.Tk = master
        self.grid(sticky="nsew")
        master.rowconfigure(0, weight=1)
        master.columnconfigure(0, weight=1)

        # --- 状态 ---
        self._image_path: Optional[str] = None
        self._img_orig: Optional[Image.Image] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._last_preview_size: Tuple[int, int] = (0, 0)
        self._engine_cache: Dict[str, Any] = {}
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._running = False
        self._cancel_requested = False
        self._result: Optional[Dict[str, Any]] = None
        self._ocr_lines: List[Any] = []
        self._preview_scale = 0.0
        self._preview_offset: Tuple[int, int] = (0, 0)
        self._highlight = -1

        self._build_ui()
        self._set_state(False)

    # ------------------------------------------------------------------
    # 界面搭建
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._build_file_row(0)
        self._build_main_area(1)
        self._build_options(2)
        self._build_ground_truth(3)
        self._build_action_row(4)
        self._build_status_bar(5)

    def _build_file_row(self, row: int) -> None:
        frame = ttk.LabelFrame(self, text=" 1. 选择图片 ", padding=6)
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="图片路径:").grid(row=0, column=0, padx=(0, 6))
        self.path_var = tk.StringVar()
        entry = ttk.Entry(frame, textvariable=self.path_var)
        entry.grid(row=0, column=1, sticky="ew")
        entry.bind("<Return>", lambda _e: self._open_image(self.path_var.get()))

        ttk.Button(frame, text="浏览…", command=self._browse).grid(
            row=0, column=2, padx=4)
        ttk.Button(frame, text="清空", command=self._clear_image).grid(
            row=0, column=3)

        self.info_var = tk.StringVar(value="尚未选择图片")
        ttk.Label(frame, textvariable=self.info_var, foreground="#666").grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))

    def _build_main_area(self, row: int) -> None:
        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.grid(row=row, column=0, sticky="nsew")

        # --- 左：预览 ---
        left = ttk.LabelFrame(paned, text=" 图片预览 ", padding=4)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self._canvas = tk.Canvas(left, background="#f4f4f4", width=self.PREVIEW_W,
                                 height=self.PREVIEW_H, highlightthickness=1,
                                 highlightbackground="#cccccc")
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._canvas.bind("<Configure>", self._on_preview_resize)
        self._canvas.bind("<Button-1>", self._on_canvas_click)

        ctrl = ttk.Frame(left)
        ctrl.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.show_boxes_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="显示识别框", variable=self.show_boxes_var,
                        command=self._draw_boxes).grid(row=0, column=0, sticky="w")
        ttk.Label(ctrl, text="点击框定位到对应行", foreground="#888").grid(
            row=1, column=0, sticky="w")
        paned.add(left, weight=0)

        # --- 右：结果 ---
        right = ttk.Frame(paned)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        notebook = ttk.Notebook(right)
        notebook.grid(row=0, column=0, sticky="nsew")

        self._build_result_tab(notebook)
        self._build_metrics_tab(notebook)
        self._build_diff_tab(notebook)
        paned.add(right, weight=1)

    def _build_result_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=4)
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)

        cols = ("no", "text", "conf")
        self.tree = ttk.Treeview(tab, columns=cols, show="headings", selectmode="browse")
        self.tree.heading("no", text="#")
        self.tree.heading("text", text="识别文本")
        self.tree.heading("conf", text="置信度")
        self.tree.column("no", width=48, anchor="center", stretch=False)
        self.tree.column("text", width=460, anchor="w")
        self.tree.column("conf", width=90, anchor="center", stretch=False)
        self.tree.tag_configure("low", background="#ffe0e0")
        self.tree.tag_configure("mid", background="#fff5d6")
        self.tree.tag_configure("high", background="#e6f7e6")
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        ys = ttk.Scrollbar(tab, orient="vertical", command=self.tree.yview)
        ys.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=ys.set)

        bar = ttk.Frame(tab)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        bar.columnconfigure(0, weight=1)
        self.avg_conf_var = tk.StringVar(value="平均置信度: —")
        ttk.Label(bar, textvariable=self.avg_conf_var, font=("", 10, "bold")).grid(
            row=0, column=0, sticky="w")
        ttk.Button(bar, text="复制识别文本", command=self._copy_text).grid(
            row=0, column=1)

        notebook.add(tab, text="  识别结果  ")

    def _build_metrics_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=10)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)

        # 顶部三张「卡片」
        cards = ttk.Frame(tab)
        cards.grid(row=0, column=0, sticky="ew")
        for i in range(3):
            cards.columnconfigure(i, weight=1)

        self.card_vars: Dict[str, tk.StringVar] = {}
        specs = [("char", "字符准确率", "#1a7f37"),
                 ("word", "词准确率", "#0969da"),
                 ("line", "行级准确率", "#8250df")]
        for col, (key, title, color) in enumerate(specs):
            box = ttk.LabelFrame(cards, text=f" {title} ", padding=8)
            box.grid(row=0, column=col, sticky="ew", padx=4)
            var = tk.StringVar(value="—")
            self.card_vars[key] = var
            ttk.Label(box, textvariable=var, font=("", 20, "bold"),
                      foreground=color, anchor="center").grid(row=0, column=0, sticky="ew")
            box.columnconfigure(0, weight=1)

        # 明细
        detail = ttk.Frame(tab)
        detail.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        detail.rowconfigure(0, weight=1)
        detail.columnconfigure(0, weight=1)

        cols = ("metric", "value")
        self.metric_tree = ttk.Treeview(detail, columns=cols, show="headings",
                                        selectmode="none")
        self.metric_tree.heading("metric", text="指标")
        self.metric_tree.heading("value", text="数值")
        self.metric_tree.column("metric", width=260, anchor="w")
        self.metric_tree.column("value", width=180, anchor="w")
        self.metric_tree.grid(row=0, column=0, sticky="nsew")
        ys = ttk.Scrollbar(detail, orient="vertical", command=self.metric_tree.yview)
        ys.grid(row=0, column=1, sticky="ns")
        self.metric_tree.configure(yscrollcommand=ys.set)

        notebook.add(tab, text="  准确率评估  ")

    def _build_diff_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=4)
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)

        cols = ("line", "ref", "hyp")
        self.diff_tree = ttk.Treeview(tab, columns=cols, show="headings",
                                      selectmode="browse")
        self.diff_tree.heading("line", text="行号")
        self.diff_tree.heading("ref", text="标注")
        self.diff_tree.heading("hyp", text="识别")
        self.diff_tree.column("line", width=52, anchor="center", stretch=False)
        self.diff_tree.column("ref", width=280, anchor="w")
        self.diff_tree.column("hyp", width=280, anchor="w")
        self.diff_tree.grid(row=0, column=0, sticky="nsew")
        ys = ttk.Scrollbar(tab, orient="vertical", command=self.diff_tree.yview)
        ys.grid(row=0, column=1, sticky="ns")
        self.diff_tree.configure(yscrollcommand=ys.set)

        self.diff_summary_var = tk.StringVar(value="填写标注文本并识别后，这里会列出差异行。")
        ttk.Label(tab, textvariable=self.diff_summary_var, foreground="#666").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        notebook.add(tab, text="  差异对照  ")

    def _build_options(self, row: int) -> None:
        frame = ttk.LabelFrame(self, text=" 2. 识别选项 ", padding=6)
        frame.grid(row=row, column=0, sticky="ew", pady=6)

        ttk.Label(frame, text="引擎:").grid(row=0, column=0, padx=(0, 4))
        self.engine_var = tk.StringVar(value="auto")
        combo = ttk.Combobox(frame, state="readonly", width=10,
                             values=[label for label, _ in ENGINE_CHOICES])
        combo.current(0)
        combo.grid(row=0, column=1, padx=(0, 14))
        self._engine_combo = combo

        ttk.Label(frame, text="检测框扩张:").grid(row=0, column=2, padx=(0, 4))
        self.unclip_var = tk.DoubleVar(value=core.DEFAULT_UNCLIP_RATIO)
        ttk.Spinbox(frame, from_=1.0, to=6.0, increment=0.1, width=6,
                    textvariable=self.unclip_var).grid(row=0, column=3, padx=(0, 14))
        ttk.Label(frame, text="（小字被裁掉行首/行尾时可调大）",
                  foreground="#888").grid(row=0, column=4, padx=(0, 14))

        ttk.Label(frame, text="放大倍数:").grid(row=1, column=0, padx=(0, 4), pady=(6, 0))
        self.scale_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(frame, from_=1.0, to=4.0, increment=0.5, width=6,
                    textvariable=self.scale_var).grid(row=1, column=1,
                                                      padx=(0, 14), pady=(6, 0))

        self.ignore_punct_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="评估时忽略标点",
                        variable=self.ignore_punct_var).grid(
            row=1, column=2, columnspan=2, sticky="w", pady=(6, 0))

        self.keep_case_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="评估时区分大小写",
                        variable=self.keep_case_var).grid(
            row=1, column=4, sticky="w", pady=(6, 0))

    def _build_ground_truth(self, row: int) -> None:
        frame = ttk.LabelFrame(
            self, text=" 3. 标注文本（可选，填写后才能计算准确率） ", padding=6)
        frame.grid(row=row, column=0, sticky="ew")
        frame.columnconfigure(0, weight=1)

        self.gt_text = tk.Text(frame, height=4, wrap="word", undo=True)
        self.gt_text.grid(row=0, column=0, sticky="ew")

        side = ttk.Frame(frame)
        side.grid(row=0, column=1, sticky="ns", padx=(6, 0))
        ttk.Button(side, text="载入文件…", command=self._load_gt_file).grid(
            row=0, column=0, sticky="ew")
        ttk.Button(side, text="清空", command=lambda: self.gt_text.delete("1.0", "end")).grid(
            row=1, column=0, sticky="ew", pady=(4, 0))

    def _build_action_row(self, row: int) -> None:
        frame = ttk.Frame(self)
        frame.grid(row=row, column=0, sticky="ew", pady=(8, 0))
        frame.columnconfigure(3, weight=1)

        self.run_btn = ttk.Button(frame, text="开始识别", command=self._start)
        self.run_btn.grid(row=0, column=0)
        self.cancel_btn = ttk.Button(frame, text="取消", command=self._cancel)
        self.cancel_btn.grid(row=0, column=1, padx=6)
        self.save_btn = ttk.Button(frame, text="导出结果…", command=self._save)
        self.save_btn.grid(row=0, column=2)

        self.progress = ttk.Progressbar(frame, mode="indeterminate", length=160)
        self.progress.grid(row=0, column=3, sticky="w", padx=12)

    def _build_status_bar(self, row: int) -> None:
        self.status_var = tk.StringVar(value="就绪")
        bar = ttk.Frame(self, relief="sunken", padding=(6, 3))
        bar.grid(row=row, column=0, sticky="ew", pady=(8, 0))
        bar.columnconfigure(0, weight=1)
        ttk.Label(bar, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        self.engine_label_var = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.engine_label_var,
                  foreground="#666").grid(row=0, column=1, sticky="e")

    # ------------------------------------------------------------------
    # 图片
    # ------------------------------------------------------------------

    def _browse(self) -> None:
        path = filedialog.askopenfilename(title="选择图片", filetypes=IMAGE_TYPES)
        if path:
            self._open_image(path)

    def _open_image(self, path: str) -> None:
        path = (path or "").strip().strip('"')
        if not path:
            return
        if not os.path.isfile(path):
            messagebox.showerror("找不到图片", f"文件不存在:\n{path}")
            return
        try:
            with Image.open(path) as img:
                img.load()
                self._img_orig = img.convert("RGB")
                fmt, size = img.format or "", img.size
        except Exception as exc:
            messagebox.showerror("无法打开图片", f"{exc}")
            return

        self._image_path = path
        self.path_var.set(path)
        self.info_var.set(f"{size[0]} × {size[1]} 像素 · {fmt} · {os.path.basename(path)}")
        self._last_preview_size = (0, 0)
        self.after(30, self._fit_preview)

    def _clear_image(self) -> None:
        self._image_path = None
        self._img_orig = None
        self._photo = None
        self._ocr_lines = []
        self._preview_scale = 0.0
        self._highlight = -1
        self.path_var.set("")
        self.info_var.set("尚未选择图片")
        self._canvas.delete("all")

    def _on_preview_resize(self, _event: tk.Event) -> None:
        self._fit_preview()

    def _fit_preview(self) -> None:
        if self._img_orig is None:
            return
        cw = max(self._canvas.winfo_width(), 60)
        ch = max(self._canvas.winfo_height(), 60)
        if abs(cw - self._last_preview_size[0]) < 6 and \
                abs(ch - self._last_preview_size[1]) < 6:
            return
        self._last_preview_size = (cw, ch)
        thumb = self._img_orig.copy()
        thumb.thumbnail((cw - 10, ch - 10), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(thumb)
        self._canvas.delete("all")
        self._canvas.create_image(cw // 2, ch // 2, image=self._photo)

        # 记录原图坐标 -> 预览坐标的变换，供画框和点击命中使用
        self._preview_scale = thumb.width / max(1, self._img_orig.width)
        self._preview_offset = ((cw - thumb.width) // 2, (ch - thumb.height) // 2)
        self._draw_boxes()

    # ---- 识别框叠加 ----

    BOX_COLORS = {"low": "#d1242f", "mid": "#bf8700", "high": "#1a7f37"}

    def _box_rect(self, box) -> Optional[Tuple[float, float, float, float]]:
        """把原图坐标的检测框换算成预览画布上的矩形。"""
        if not box or not self._preview_scale:
            return None
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        sx, sy = self._preview_scale, self._preview_scale
        ox, oy = self._preview_offset
        return (ox + min(xs) * sx, oy + min(ys) * sy,
                ox + max(xs) * sx, oy + max(ys) * sy)

    def _draw_boxes(self) -> None:
        self._canvas.delete("box")
        if not self.show_boxes_var.get() or not self._preview_scale:
            return
        for i, line in enumerate(self._ocr_lines):
            rect = self._box_rect(getattr(line, "box", None))
            if rect is None:
                continue
            x0, y0, x1, y1 = rect
            level = "low" if line.score < 0.7 else ("mid" if line.score < 0.9 else "high")
            color = self.BOX_COLORS[level]
            self._canvas.create_rectangle(
                x0, y0, x1, y1, outline=color, width=3 if i == self._highlight else 1,
                tags=("box", f"b{i}"))

    def _on_canvas_click(self, event: tk.Event) -> None:
        """点击预览图上的识别框，选中右侧对应行。"""
        if not self._ocr_lines:
            return
        for i in range(len(self._ocr_lines) - 1, -1, -1):
            rect = self._box_rect(getattr(self._ocr_lines[i], "box", None))
            if rect and rect[0] <= event.x <= rect[2] and rect[1] <= event.y <= rect[3]:
                self._select_line(i)
                return

    def _on_tree_select(self, _event: tk.Event) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        self._highlight = self.tree.index(sel[0])
        self._draw_boxes()

    def _select_line(self, index: int) -> None:
        items = self.tree.get_children()
        if not (0 <= index < len(items)):
            return
        self.tree.selection_set(items[index])
        self.tree.see(items[index])
        self._highlight = index
        self._draw_boxes()

    def _load_gt_file(self) -> None:
        path = filedialog.askopenfilename(
            title="选择标注文本文件",
            filetypes=[("文本文件", "*.txt *.md *.csv"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            for enc in ("utf-8-sig", "utf-8", "gbk"):
                try:
                    with open(path, "r", encoding=enc) as fp:
                        content = fp.read()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise UnicodeDecodeError("unknown", b"", 0, 1, "无法识别文件编码")
        except Exception as exc:
            messagebox.showerror("读取失败", f"{exc}")
            return
        self.gt_text.delete("1.0", "end")
        self.gt_text.insert("1.0", content)

    # ------------------------------------------------------------------
    # 识别流程
    # ------------------------------------------------------------------

    def _selected_engine_key(self) -> str:
        idx = self._engine_combo.current()
        if idx < 0:
            return "auto"
        return ENGINE_CHOICES[idx][1]

    def _get_engine(self, key: str):
        """引擎加载较慢，按引擎名缓存复用。"""
        if key not in self._engine_cache:
            self._engine_cache[key] = core.build_engine(key)
        return self._engine_cache[key]

    def _start(self) -> None:
        if self._running:
            return
        if not self._image_path:
            messagebox.showinfo("请先选择图片", "请点击「浏览…」选择一张图片。")
            return
        try:
            unclip = float(self.unclip_var.get())
            scale = float(self.scale_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("参数错误", "检测框扩张 / 放大倍数必须是数字。")
            return
        if scale < 1.0:
            messagebox.showerror("参数错误", "放大倍数不能小于 1。")
            return

        gt = self.gt_text.get("1.0", "end").strip()

        self._running = True
        self._cancel_requested = False
        self._set_state(True)
        self.status_var.set("正在加载引擎并识别，请稍候…")
        self.progress.start(12)

        params = {
            "image": self._image_path,
            "engine": self._selected_engine_key(),
            "unclip": unclip,
            "scale": scale,
            "gt": gt or None,
            "ignore_punct": self.ignore_punct_var.get(),
            "keep_case": self.keep_case_var.get(),
        }
        threading.Thread(target=self._worker, args=(params,), daemon=True).start()
        self.after(80, self._poll)

    def _worker(self, p: Dict[str, Any]) -> None:
        tmp_path = None
        try:
            engine = self._get_engine(p["engine"])
            if hasattr(engine, "set_unclip_ratio"):
                engine.set_unclip_ratio(p["unclip"])

            work_path, tmp_path = core.prepare_image(p["image"], p["scale"])
            if self._cancel_requested:
                return
            ocr_lines = engine.run_full(work_path)
            if p["scale"] != 1.0:
                # 检测框在放大后的图上算出，换算回原图坐标才能叠到预览上
                ocr_lines = [
                    ln._replace(box=[(x / p["scale"], y / p["scale"])
                                     for x, y in ln.box] if ln.box else None)
                    for ln in ocr_lines
                ]

            recognized = "\n".join(ln.text for ln in ocr_lines)
            metrics = None
            if p["gt"] is not None:
                metrics = core.evaluate(p["gt"], recognized,
                                        ignore_punct=p["ignore_punct"],
                                        keep_case=p["keep_case"])
            if self._cancel_requested:
                return
            self._queue.put({
                "ok": True,
                "lines": ocr_lines,
                "metrics": metrics,
                "recognized": recognized,
                "gt": p["gt"],
                "engine": engine.name,
                "unclip": p["unclip"],
                "scale": p["scale"],
                "image": p["image"],
            })
        except Exception as exc:
            self._queue.put({"ok": False, "error": str(exc),
                             "trace": traceback.format_exc()})
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _poll(self) -> None:
        try:
            result = self._queue.get_nowait()
        except queue.Empty:
            if self._running:
                self.after(80, self._poll)
            return

        self._running = False
        self.progress.stop()
        self._set_state(False)

        if not result.get("ok"):
            self.status_var.set("识别失败")
            self.engine_label_var.set("")
            messagebox.showerror("识别失败", result.get("error", "未知错误"))
            return

        self._result = result
        self._render(result)

    def _cancel(self) -> None:
        if not self._running:
            return
        self._cancel_requested = True
        self._running = False
        self.progress.stop()
        self._set_state(False)
        self.status_var.set("已取消（后台任务会在当前步骤结束后停止）")

    def _set_state(self, running: bool) -> None:
        self.run_btn.configure(state="disabled" if running else "normal")
        self.save_btn.configure(state="disabled" if (running or not self._result)
                                else "normal")
        self.cancel_btn.configure(state="normal" if running else "disabled")

    # ------------------------------------------------------------------
    # 结果渲染
    # ------------------------------------------------------------------

    def _render(self, r: Dict[str, Any]) -> None:
        lines = r["lines"]
        metrics: Optional[Dict[str, Any]] = r["metrics"]

        self._render_lines(lines)
        self._render_metrics(metrics)
        self._render_diff(r["gt"], r["recognized"], metrics)

        with_box = sum(1 for ln in lines if getattr(ln, "box", None))
        bits = [f"识别到 {len(lines)} 行", f"引擎 {r['engine']}",
                f"unclip={r['unclip']:g}"]
        if with_box:
            bits.append(f"{with_box} 个检测框")
        if r["scale"] != 1.0:
            bits.append(f"放大 {r['scale']:g}x")
        bits.append("已计算准确率" if metrics else "未提供标注文本")
        self.status_var.set(" · ".join(bits))
        self.engine_label_var.set(f"检测框扩张系数 {r['unclip']:g}")

    def _render_lines(self, lines: List[Any]) -> None:
        self.tree.delete(*self.tree.get_children())
        self._ocr_lines = list(lines)
        self._highlight = -1
        for i, line in enumerate(lines, 1):
            tag = ("low" if line.score < 0.7
                   else ("mid" if line.score < 0.9 else "high"))
            self.tree.insert("", "end",
                             values=(i, line.text, f"{line.score * 100:.2f}%"),
                             tags=(tag,))
        if lines:
            avg = sum(ln.score for ln in lines) / len(lines)
            self.avg_conf_var.set(
                f"平均置信度: {avg * 100:.2f}%（{len(lines)} 行 · "
                f"绿色 ≥90%，黄色 ≥70%，红色 <70%）")
        else:
            self.avg_conf_var.set("平均置信度: —（未识别出文字）")
        self._draw_boxes()

    def _render_metrics(self, m: Optional[Dict[str, Any]]) -> None:
        self.metric_tree.delete(*self.metric_tree.get_children())
        if m is None:
            for var in self.card_vars.values():
                var.set("—")
            self.metric_tree.insert("", "end", values=(
                "提示", "填写标注文本后即可计算准确率"))
            return

        self.card_vars["char"].set(f"{m['char_accuracy'] * 100:.2f}%")
        self.card_vars["word"].set(f"{m['word_accuracy'] * 100:.2f}%")
        self.card_vars["line"].set(f"{m['line_accuracy'] * 100:.2f}%")

        rows = [
            ("标注字符数 / 识别字符数", f"{m['ref_chars']} / {m['hyp_chars']}"),
            ("字符编辑距离", str(m["char_edit_distance"])),
            ("字符错误率 CER", f"{m['cer'] * 100:.2f}%"),
            ("词数（标注 / 识别）", f"{m['ref_words']} / {m['hyp_words']}"),
            ("词错误率 WER", f"{m['wer'] * 100:.2f}%"),
            ("序列相似度", f"{m['sequence_ratio'] * 100:.2f}%"),
            ("行级命中", f"{m['line_hit']} / {m['line_total']} 行完全一致"),
            ("整段完全匹配", "是" if m["exact_match"] else "否"),
        ]
        for name, value in rows:
            self.metric_tree.insert("", "end", values=(name, value))

    def _render_diff(self, gt: Optional[str], recognized: str,
                     metrics: Optional[Dict[str, Any]]) -> None:
        self.diff_tree.delete(*self.diff_tree.get_children())
        if gt is None:
            self.diff_summary_var.set(
                "未提供标注文本。填写后此处会列出「标注 vs 识别」不一致的行。")
            return

        ref_lines = core.split_lines(gt)
        hyp_lines = core.split_lines(recognized)
        count = 0
        for i in range(max(len(ref_lines), len(hyp_lines))):
            ref = ref_lines[i] if i < len(ref_lines) else "(无此行)"
            hyp = hyp_lines[i] if i < len(hyp_lines) else "(无此行)"
            if ref != hyp:
                count += 1
                self.diff_tree.insert("", "end", values=(i + 1, ref, hyp))

        if count == 0:
            self.diff_summary_var.set("所有行完全一致 ✓")
        else:
            extra = ""
            if metrics is not None:
                extra = (f"　字符准确率 {metrics['char_accuracy'] * 100:.2f}%，"
                         f"词准确率 {metrics['word_accuracy'] * 100:.2f}%")
            self.diff_summary_var.set(
                f"共 {count} 行存在差异（按行号顺序比对，对换行切分敏感）。{extra}")

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------

    def _copy_text(self) -> None:
        if not self._result:
            messagebox.showinfo("暂无结果", "请先执行识别。")
            return
        self.master.clipboard_clear()
        self.master.clipboard_append(self._result["recognized"])
        self.status_var.set("识别文本已复制到剪贴板")

    def _save(self) -> None:
        if not self._result:
            messagebox.showinfo("暂无结果", "请先执行识别。")
            return
        r = self._result
        base = os.path.splitext(os.path.basename(r["image"]))[0] or "ocr"
        path = filedialog.asksaveasfilename(
            title="导出结果", defaultextension=".txt",
            initialfile=f"{base}_ocr.txt",
            filetypes=[("文本文件", "*.txt"), ("JSON 文件", "*.json")])
        if not path:
            return
        try:
            if path.lower().endswith(".json"):
                payload = {
                    "image": r["image"],
                    "engine": r["engine"],
                    "unclip_ratio": r["unclip"],
                    "scale": r["scale"],
                    "lines": [
                        {"text": ln.text, "confidence": ln.score,
                         "box": [[round(x, 1), round(y, 1)] for x, y in ln.box]
                                if getattr(ln, "box", None) else None}
                        for ln in r["lines"]
                    ],
                    "recognized_text": r["recognized"],
                    "average_confidence": (
                        sum(ln.score for ln in r["lines"]) / len(r["lines"])
                        if r["lines"] else 0.0),
                    "ground_truth": r["gt"],
                    "metrics": r["metrics"],
                }
                with open(path, "w", encoding="utf-8") as fp:
                    json.dump(payload, fp, ensure_ascii=False, indent=2)
            else:
                with open(path, "w", encoding="utf-8") as fp:
                    fp.write("识别结果\n" + "=" * 40 + "\n")
                    for i, ln in enumerate(r["lines"], 1):
                        fp.write(f"{i:>4}. {ln.text}   "
                                 f"[置信度 {ln.score * 100:.2f}%]\n")
                    if r["lines"]:
                        avg = sum(ln.score for ln in r["lines"]) / len(r["lines"])
                        fp.write(f"\n平均置信度: {avg * 100:.2f}%\n")
                    if r["metrics"]:
                        m = r["metrics"]
                        fp.write("\n准确率评估\n" + "=" * 40 + "\n")
                        fp.write(f"字符准确率 (1-CER): {m['char_accuracy'] * 100:.2f}%"
                                 f"  (CER = {m['cer'] * 100:.2f}%)\n")
                        fp.write(f"词准确率   (1-WER): {m['word_accuracy'] * 100:.2f}%"
                                 f"  (WER = {m['wer'] * 100:.2f}%)\n")
                        fp.write(f"序列相似度        : {m['sequence_ratio'] * 100:.2f}%\n")
                        fp.write(f"行级准确率        : {m['line_accuracy'] * 100:.2f}%"
                                 f"  ({m['line_hit']}/{m['line_total']} 行完全一致)\n")
                        fp.write(f"完全匹配          : {'是' if m['exact_match'] else '否'}\n")
        except Exception as exc:
            messagebox.showerror("导出失败", f"{exc}")
            return
        self.status_var.set(f"已导出: {path}")


# ---------------------------------------------------------------------------


def main() -> int:
    setup_dpi_awareness()

    root = tk.Tk()
    root.title("图片文字识别 (OCR) 与准确率评估")
    root.minsize(940, 680)

    # 中文界面下 ttk 默认主题更协调
    try:
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass

    try:
        family = pick_ui_font(root)
        root.option_add("*Font", (family, 10))
    except Exception:
        pass

    OcrApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    # pythonw 下没有控制台，print 到 stderr 会报错，这里兜底
    if sys.stderr is None:
        import io

        sys.stderr = io.StringIO()
    if sys.stdout is None:
        import io

        sys.stdout = io.StringIO()
    sys.exit(main())
