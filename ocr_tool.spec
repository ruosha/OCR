# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置：一次生成两个可执行文件
  - OCR-CLI.exe   命令行版（带控制台窗口）
  - OCR-GUI.exe   图形界面版（无控制台窗口）

用法:
    python -m PyInstaller --clean --noconfirm ocr_tool.spec

产物位于 dist/ 目录。默认打成单文件（onefile），分发时每个 exe 都可独立运行。
"""
import os

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

HERE = SPECPATH

# ---------------------------------------------------------------- 依赖收集
# RapidOCR 的 config.yaml 与 models/*.onnx 必须一并打包，否则运行时报找不到模型
ocr_datas = collect_data_files("rapidocr_onnxruntime")

# 这些子包是引擎内部用 importlib.import_module(字符串) 动态加载的
# （如 'ch_ppocr_v3_det'），PyInstaller 的静态分析发现不了，必须显式声明
ocr_hidden = collect_submodules("rapidocr_onnxruntime")

# onnxruntime 的原生 DLL
ort_bins = collect_dynamic_libs("onnxruntime")

# 排除明显用不到的重型依赖，显著缩小体积
# 注意：不要排除 tcl/tk —— tkinter 依赖它们，排除后界面会直接起不来
EXCLUDES = [
    "matplotlib", "scipy", "pandas", "IPython", "jupyter", "notebook",
    "nbconvert", "nbformat", "pytest", "PyQt5", "PyQt6", "PySide2",
    "PySide6", "wx", "sphinx", "docutils", "pydoc_data",
    # 以下为本项目确实用不到的库内子模块
    "numpy.f2py", "numpy.distutils",
    "PIL.ImageQt", "PIL.ImageGrab", "PIL.ImageShow",
    "onnxruntime.tools", "onnxruntime.transformers",
    "onnxruntime.quantization",
    "setuptools", "pkg_resources", "lib2to3",
]

# 从产物中剔除体积大但确定用不到的原生文件。
# 实测 opencv_videoio_ffmpeg*.dll 有 29 MB（压缩后仍占 13 MB），
# 它只用于视频解码，而本工具只处理静态图片。
BINARY_DROP_KEYS = (
    "opencv_videoio_ffmpeg",
)

# 同理，剔除库自带的测试数据
DATAS_DROP_KEYS = (
    "numpy\\tests", "numpy/tests",
    "numpy\\testing", "numpy/testing",
    "PIL\\tests", "PIL/tests",
)


def _slim(items, keep_key):
    """按关键词过滤 PyInstaller 的 (name, path, typecode) 三元组列表。"""
    kept = []
    dropped = []
    for item in items:
        name = item[0].replace("/", "\\").lower()
        if any(k.replace("/", "\\").lower() in name for k in keep_key):
            dropped.append(item)
        else:
            kept.append(item)
    return kept, dropped


_COMMON = dict(
    pathex=[HERE],
    binaries=ort_bins,
    hiddenimports=ocr_hidden,
    excludes=EXCLUDES,
    noarchive=False,
)


def _slim_analysis(a, label):
    """把分析结果里不需要的大文件剔除掉，并打印剔除了什么。"""
    a.binaries, dropped_b = _slim(a.binaries, BINARY_DROP_KEYS)
    a.datas, dropped_d = _slim(a.datas, DATAS_DROP_KEYS)
    total = sum(os.path.getsize(p) for _, p, _ in dropped_b + dropped_d
                if os.path.exists(p))
    print(f"[{label}] 剔除 {len(dropped_b)} 个二进制 + {len(dropped_d)} 项数据，"
          f"原始体积约 {total / 1024 / 1024:.1f} MB")
    for name, _, _ in dropped_b:
        print(f"         - {name}")
    return a

# ---------------------------------------------------------------- 命令行版
a_cli = _slim_analysis(
    Analysis([os.path.join(HERE, "ocr_cli.py")], datas=ocr_datas, **_COMMON),
    "OCR-CLI",
)
pyz_cli = PYZ(a_cli.pure)

exe_cli = EXE(
    pyz_cli,
    a_cli.scripts,
    a_cli.binaries,
    a_cli.datas,
    [],
    name="OCR-CLI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)

# ---------------------------------------------------------------- 图形界面版
# ocr_gui.py 通过 importlib 按路径加载 ocr_cli.py，所以 ocr_cli.py 必须作为数据文件带上
gui_datas = ocr_datas + [(os.path.join(HERE, "ocr_cli.py"), ".")]

a_gui = _slim_analysis(
    Analysis([os.path.join(HERE, "ocr_gui.py")], datas=gui_datas, **_COMMON),
    "OCR-GUI",
)
pyz_gui = PYZ(a_gui.pure)

exe_gui = EXE(
    pyz_gui,
    a_gui.scripts,
    a_gui.binaries,
    a_gui.datas,
    [],
    name="OCR-GUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)
