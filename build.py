# -*- coding: utf-8 -*-
"""
一键打包成 exe
==============

用法：

    双击 build.bat        （推荐，Windows 用户直接双击即可）
    或   python build.py

流程：检查依赖 -> 缺失则自动安装 -> 调用 ocr_tool.spec 打包 -> 报告产物。

为什么中文提示放在这里而不是 build.bat 里：
    cmd.exe 按系统 OEM 代码页（中文 Windows 是 GBK）读取 .bat 文件，
    而本项目源码是 UTF-8。如果把中文写进 .bat，会被错解成乱码，
    严重时错解的字节还会破坏行结构、把普通文本当成命令执行
    （实测出现过把 `dist\\OCR-CLI.exe` 当命令跑掉的情况）。
    所以 .bat 只保留纯 ASCII 的最小启动逻辑，其余都在这里做。
"""

from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MIRROR = ["-i", "https://pypi.tuna.tsinghua.edu.cn/simple"]

# 控制台编码兜底：GBK 下打印不了 ★ 等符号，不应因此让脚本崩溃
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

# (import 名, pip 包名, 显示名)
DEPENDENCIES = [
    ("rapidocr_onnxruntime", "rapidocr-onnxruntime", "rapidocr-onnxruntime（OCR 引擎）"),
    ("PIL", "pillow", "pillow（图片处理）"),
    ("PyInstaller", "pyinstaller", "pyinstaller（打包工具）"),
]


def section(title: str) -> None:
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


def has_module(name: str) -> bool:
    return subprocess.call(
        [sys.executable, "-c", f"import {name}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def ensure(module: str, package: str, label: str) -> bool:
    if has_module(module):
        print(f"      {label}  [已安装]")
        return True
    print(f"      缺少 {label}，正在安装…")
    rc = subprocess.call([sys.executable, "-m", "pip", "install",
                          *MIRROR, package])
    if rc != 0:
        print(f"      [失败] {label} 安装失败，请检查网络后重试")
        return False
    return True


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="把本项目打包成免安装 exe")
    parser.add_argument("--no-open", action="store_true",
                        help="打包完成后不自动打开 dist 目录")
    parser.add_argument("--skip-deps", action="store_true",
                        help="跳过依赖检查（确认已装好依赖时可用）")
    args = parser.parse_args(argv)

    section("图片文字识别工具 — 一键打包")
    print(f"  Python : {sys.version.split()[0]}")
    print(f"  项目   : {HERE}")

    spec = os.path.join(HERE, "ocr_tool.spec")
    if not os.path.isfile(spec):
        print(f"\n[错误] 找不到打包配置: {spec}")
        print("       请在项目目录下运行本脚本。")
        return 1

    if not args.skip_deps:
        print("\n[1/2] 检查依赖")
        for module, package, label in DEPENDENCIES:
            if not ensure(module, package, label):
                return 1
    else:
        print("\n[1/2] 已跳过依赖检查")

    print("\n[2/2] 开始打包（约需 1 分钟，请稍候）\n")
    rc = subprocess.call([sys.executable, "-m", "PyInstaller",
                          "--clean", "--noconfirm", spec], cwd=HERE)
    if rc != 0:
        section("打包失败")
        print("  请查看上方 PyInstaller 日志定位原因。")
        return rc if rc > 0 else 1

    dist = os.path.join(HERE, "dist")
    section("打包完成")
    print(f"  产物目录: {dist}\n")
    for name, desc in (("OCR-CLI.exe", "命令行版"),
                       ("OCR-GUI.exe", "图形界面版，双击即用")):
        path = os.path.join(dist, name)
        if os.path.isfile(path):
            size = os.path.getsize(path) / 1024 / 1024
            print(f"    {name:<14}{size:7.1f} MB   {desc}")
        else:
            print(f"    {name:<14}  [缺失]")
    print("\n  两个 exe 都是单文件、模型已内置，")
    print("  可直接拷到没有装 Python 的 Windows 电脑运行。")

    if not args.no_open and os.path.isdir(dist):
        try:
            os.startfile(dist)  # noqa: S606  (Windows only)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
