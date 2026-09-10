# -*- coding: utf-8 -*-
"""
图片文字识别 (OCR) 程序 + 识别准确率评估
=================================================

功能:
  1. 识别图片中的文字（默认使用 RapidOCR / ONNXRuntime 引擎，纯 Python，无需额外安装外部程序）
  2. 输出识别结果以及每一行的置信度
  3. 与标注文本 (ground truth) 对比，输出识别准确率:
       - 字符准确率 (1 - CER，基于编辑距离)
       - 词准确率   (1 - WER，基于词序列编辑距离)
       - 完全匹配    (整段文本是否完全一致)
       - 行级准确率  (逐行完全一致的占比)
       - 平均置信度  (OCR 引擎自身给出的置信度均值)

用法:
  python ocr_cli.py 图片.png                      # 仅识别
  python ocr_cli.py 图片.png --gt 标注.txt        # 识别并与标注文件对比
  python ocr_cli.py 图片.png --gt "预期文本"      # 识别并与给定文本对比
  python ocr_cli.py --demo                        # 自动生成测试图片并评估准确率
  python ocr_cli.py --demo --save demo.png        # 生成测试图片时同时保存图片

  # 可选调优（默认已按实测调好，通常不需要改）
  python ocr_cli.py 图片.png --unclip 2.5         # 检测框扩张系数，防止裁掉行首/行尾字符
  python ocr_cli.py 小图.png --scale 2            # 仅对低分辨率图片放大后再识别

依赖:
  pip install rapidocr-onnxruntime     # 推荐（默认引擎）
  pip install pillow                   # 仅 --demo 生成测试图片时需要
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unicodedata
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 一、通用文本工具
# ---------------------------------------------------------------------------


def levenshtein(a: Sequence[Any], b: Sequence[Any]) -> int:
    """编辑距离，支持字符串或任意可比较元素组成的序列（如词表）。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,        # 删除
                    current[j - 1] + 1,     # 插入
                    previous[j - 1] + (ca != cb),  # 替换
                )
            )
        previous = current
    return previous[-1]


def is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF      # CJK 统一汉字
        or 0x3400 <= code <= 0x4DBF   # 扩展 A
        or 0x3040 <= code <= 0x30FF   # 日文假名
        or 0xAC00 <= code <= 0xD7AF   # 韩文
    )


def normalize(
    text: str,
    keep_case: bool = False,
    ignore_punct: bool = False,
) -> str:
    """文本归一化：全角转半角、去除空白、可选去标点/忽略大小写。"""
    text = unicodedata.normalize("NFKC", text)
    if not keep_case:
        text = text.lower()
    out: List[str] = []
    for ch in text:
        if ch.isspace():
            continue
        if ignore_punct and unicodedata.category(ch).startswith(("P", "S")):
            continue
        out.append(ch)
    return "".join(out)


def tokenize(text: str) -> List[str]:
    """切词：CJK 字符逐字成词，连续的字母/数字合并为一个词，其余为分隔符。"""
    tokens: List[str] = []
    buf: List[str] = []
    for ch in text:
        if is_cjk(ch):
            if buf:
                tokens.append("".join(buf))
                buf = []
            tokens.append(ch)
        elif ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                tokens.append("".join(buf))
                buf = []
    if buf:
        tokens.append("".join(buf))
    return tokens


def split_lines(text: str) -> List[str]:
    return [ln for ln in (l.strip() for l in text.splitlines()) if ln]


# 放大后的长边上限：超过这个尺寸 OCR 反而更容易出错（实测 4096px 原图放大 2x 后
# 识别行数从 160 掉到 63），因此给出警告而不是静默产生更差的结果。
MAX_OCR_SIDE = 10000


def prepare_image(image_path: str, scale: float) -> Tuple[str, Optional[str]]:
    """按倍数放大图片后交给 OCR，返回 (实际识别路径, 需要清理的临时文件)。"""
    if scale == 1.0:
        return image_path, None

    from PIL import Image

    with Image.open(image_path) as img:
        size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
        if max(size) > MAX_OCR_SIDE:
            suggest = MAX_OCR_SIDE / max(img.width, img.height)
            print(
                f"[警告] 原图 {img.width}x{img.height} 放大 {scale:g}x 后达 "
                f"{size[0]}x{size[1]}，超过 {MAX_OCR_SIDE}px 上限。\n"
                f"        过大的输入不会提升识别率，反而可能变差；建议 --scale <= {suggest:.2f}",
                file=sys.stderr,
            )
        resized = img.convert("RGB").resize(size, Image.LANCZOS)
        fd, tmp_path = tempfile.mkstemp(prefix="ocr_scaled_", suffix=".png")
        os.close(fd)
        resized.save(tmp_path)
    return tmp_path, tmp_path


# ---------------------------------------------------------------------------
# 二、准确率评估
# ---------------------------------------------------------------------------


def evaluate(
    reference: str,
    hypothesis: str,
    ignore_punct: bool = False,
    keep_case: bool = False,
) -> Dict[str, Any]:
    """计算识别准确率指标。"""
    ref = normalize(reference, keep_case=keep_case, ignore_punct=ignore_punct)
    hyp = normalize(hypothesis, keep_case=keep_case, ignore_punct=ignore_punct)

    # --- 字符级 ---
    char_dist = levenshtein(ref, hyp)
    cer = char_dist / len(ref) if ref else (0.0 if not hyp else 1.0)
    char_accuracy = max(0.0, 1.0 - cer)

    # --- 词级 ---
    ref_tokens, hyp_tokens = tokenize(ref), tokenize(hyp)
    word_dist = levenshtein(ref_tokens, hyp_tokens)
    wer = word_dist / len(ref_tokens) if ref_tokens else (0.0 if not hyp_tokens else 1.0)
    word_accuracy = max(0.0, 1.0 - wer)

    # --- 完全匹配 ---
    exact_match = ref == hyp

    # --- 行级 ---
    ref_lines = [normalize(l, keep_case=keep_case, ignore_punct=ignore_punct)
                 for l in split_lines(reference)]
    hyp_lines = [normalize(l, keep_case=keep_case, ignore_punct=ignore_punct)
                 for l in split_lines(hypothesis)]
    line_total = max(len(ref_lines), len(hyp_lines))
    line_hit = sum(
        1 for i in range(min(len(ref_lines), len(hyp_lines)))
        if ref_lines[i] == hyp_lines[i]
    )
    line_accuracy = (line_hit / line_total) if line_total else 0.0

    return {
        "ref_chars": len(ref),
        "hyp_chars": len(hyp),
        "char_edit_distance": char_dist,
        "cer": cer,
        "char_accuracy": char_accuracy,
        "ref_words": len(ref_tokens),
        "hyp_words": len(hyp_tokens),
        "word_edit_distance": word_dist,
        "wer": wer,
        "word_accuracy": word_accuracy,
        "exact_match": exact_match,
        "line_total": line_total,
        "line_hit": line_hit,
        "line_accuracy": line_accuracy,
        "sequence_ratio": SequenceRatio(ref, hyp),
    }


def SequenceRatio(a: str, b: str) -> float:
    """difflib 序列相似度（辅助参考指标）。"""
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# 三、OCR 引擎封装
# ---------------------------------------------------------------------------


class OcrLine(NamedTuple):
    """一行识别结果。box 为 4 个顶点的多边形（原图像素坐标），拿不到时为 None。"""

    text: str
    score: float
    box: Optional[List[Tuple[float, float]]] = None


class OcrEngine:
    """统一的 OCR 引擎接口。

    - run_full() 返回 OcrLine 列表（含检测框坐标）
    - run()      仅返回 (文本, 置信度)，保持向后兼容
    """

    name = "unknown"

    def run_full(self, image_path: str) -> List[OcrLine]:  # pragma: no cover
        raise NotImplementedError

    def run(self, image_path: str) -> List[Tuple[str, float]]:
        return [(line.text, line.score) for line in self.run_full(image_path)]


# 检测框扩张系数。官方默认值 1.6 会让紧贴文字的框裁掉行首/行尾字符（实测带
# 【】（）的文本行准确率仅 96.2%）。实测 2.5 在密集排版海报上整行精确命中
# 44/116 -> 50/116，且在合成验证集上边缘符号用例 96.2% -> 100%（详见 README）。
DEFAULT_UNCLIP_RATIO = 2.5


class RapidOcrEngine(OcrEngine):
    name = "RapidOCR (ONNXRuntime)"

    def __init__(self, unclip_ratio: float = DEFAULT_UNCLIP_RATIO) -> None:
        self._api = None
        self._engine = None
        self._unclip = unclip_ratio
        try:
            from rapidocr_onnxruntime import RapidOCR  # 1.x 版本

            self._api = "v1"
        except ImportError:
            try:
                from rapidocr import RapidOCR  # 2.x 版本

                self._api = "v2"
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "未安装 RapidOCR，请执行: pip install rapidocr-onnxruntime"
                ) from exc
        self._engine = RapidOCR()

    def _call(self, image_path: str):
        """按引擎版本调用，1.x 支持传入检测参数。"""
        if self._api == "v1" and self._unclip is not None:
            try:
                return self._engine(image_path, unclip_ratio=self._unclip)
            except (TypeError, ValueError):
                pass
        return self._engine(image_path)

    def set_unclip_ratio(self, value: float) -> None:
        """调整检测框扩张系数（供调用方动态调节，无需重建引擎）。"""
        self._unclip = value

    @staticmethod
    def _parse_box(raw: Any) -> Optional[List[Tuple[float, float]]]:
        """把引擎返回的多边形转成 [(x, y), ...]，失败时返回 None。"""
        if raw is None:
            return None
        try:
            pts = [(float(p[0]), float(p[1])) for p in raw]
        except (TypeError, IndexError, ValueError):
            return None
        return pts if len(pts) >= 3 else None

    def run_full(self, image_path: str) -> List[OcrLine]:
        if self._api == "v1":
            output = self._call(image_path)
            result = output[0] if isinstance(output, tuple) else output
            if not result:
                return []
            lines: List[OcrLine] = []
            for item in result:
                # item = [box, text, score]
                text = str(item[1]).strip()
                if not text:
                    continue
                score = float(item[2]) if len(item) > 2 and item[2] is not None else 0.0
                box = self._parse_box(item[0]) if len(item) > 0 else None
                lines.append(OcrLine(text, score, box))
            return lines

        # v2 API
        result = self._call(image_path)
        texts = list(getattr(result, "txts", None) or [])
        scores = list(getattr(result, "scores", None) or [])
        boxes = list(getattr(result, "boxes", None) or [])
        lines = []
        for i, text in enumerate(texts):
            text = str(text).strip()
            if not text:
                continue
            score = float(scores[i]) if i < len(scores) and scores[i] is not None else 0.0
            box = self._parse_box(boxes[i]) if i < len(boxes) else None
            lines.append(OcrLine(text, score, box))
        return lines


class TesseractEngine(OcrEngine):
    name = "Tesseract (pytesseract)"

    def __init__(self) -> None:
        import pytesseract  # noqa: F401

        self._pytesseract = pytesseract

    def run_full(self, image_path: str) -> List[OcrLine]:
        from PIL import Image

        data = self._pytesseract.image_to_data(
            Image.open(image_path), lang="chi_sim+eng", output_type="dict"
        )
        # 按 (块, 段, 行) 分组，把同一行的词拼成一行文本，置信度取该行均值
        grouped: "Dict[Tuple[int, int, int], List[Tuple[str, float, int, int, int, int]]]" = {}
        order: List[Tuple[int, int, int]] = []
        for i, raw_text in enumerate(data.get("text", [])):
            text = (raw_text or "").strip()
            if not text:
                continue
            raw_conf = data.get("conf", [])[i] if i < len(data.get("conf", [])) else "-1"
            try:
                conf = max(0.0, float(raw_conf)) / 100.0
            except (TypeError, ValueError):
                conf = 0.0

            def _int(key: str) -> int:
                values = data.get(key) or []
                try:
                    return int(values[i])
                except (IndexError, TypeError, ValueError):
                    return 0

            key = (_int("block_num"), _int("par_num"), _int("line_num"))
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append((text, conf, _int("left"), _int("top"),
                                 _int("width"), _int("height")))

        lines: List[OcrLine] = []
        for key in order:
            words = grouped[key]
            joined = "".join(w[0] for w in words) if any(is_cjk(w[0][0]) for w in words) \
                else " ".join(w[0] for w in words)
            if not joined:
                continue
            avg = sum(w[1] for w in words) / len(words)
            x0 = min(w[2] for w in words)
            y0 = min(w[3] for w in words)
            x1 = max(w[2] + w[4] for w in words)
            y1 = max(w[3] + w[5] for w in words)
            box = [(float(x0), float(y0)), (float(x1), float(y0)),
                   (float(x1), float(y1)), (float(x0), float(y1))]
            lines.append(OcrLine(joined, avg, box))
        return lines


def build_engine(prefer: str = "auto",
                 unclip_ratio: float = DEFAULT_UNCLIP_RATIO) -> OcrEngine:
    """按优先级创建可用的 OCR 引擎。"""
    if prefer in ("auto", "rapidocr"):
        try:
            return RapidOcrEngine(unclip_ratio=unclip_ratio)
        except Exception as exc:
            if prefer == "rapidocr":
                raise
            print(f"[警告] RapidOCR 不可用: {exc}", file=sys.stderr)

    if prefer in ("auto", "pytesseract"):
        try:
            return TesseractEngine()
        except Exception as exc:
            if prefer == "pytesseract":
                raise
            print(f"[警告] Tesseract 不可用: {exc}", file=sys.stderr)

    raise RuntimeError(
        "没有可用的 OCR 引擎，请安装其中一个:\n"
        "  pip install rapidocr-onnxruntime\n"
        "  pip install pytesseract  (还需要安装 Tesseract 程序)"
    )


# ---------------------------------------------------------------------------
# 四、生成演示图片（用于自测准确率）
# ---------------------------------------------------------------------------

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhl.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/PingFang.ttc",
]

DEMO_LINES = [
    "图片文字识别测试 2026-09-10",
    "The quick brown fox jumps over the lazy dog",
    "订单编号：A1B2C3D4  金额：￥1234.56",
    "准确率评估 Accuracy = 98.75%",
]


def make_demo_image(path: str, lines: Optional[List[str]] = None,
                    font_size: int = 38) -> List[str]:
    """生成一张白底黑字的测试图片，返回图片中写入的文本行。"""
    from PIL import Image, ImageDraw, ImageFont

    lines = lines or DEMO_LINES

    font = None
    for candidate in FONT_CANDIDATES:
        if os.path.exists(candidate):
            try:
                font = ImageFont.truetype(candidate, font_size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    padding, line_gap = 40, 24
    probe = Image.new("RGB", (10, 10), "white")
    draw = ImageDraw.Draw(probe)
    widths, heights = [], []
    for line in lines:
        box = draw.textbbox((0, 0), line, font=font)
        widths.append(box[2] - box[0])
        heights.append(box[3] - box[1])

    width = max(widths) + padding * 2
    height = sum(heights) + line_gap * (len(lines) - 1) + padding * 2
    image = Image.new("RGB", (int(width), int(height)), "white")
    draw = ImageDraw.Draw(image)

    y = padding
    for line, h in zip(lines, heights):
        draw.text((padding, y), line, fill="black", font=font)
        y += h + line_gap

    image.save(path)
    return lines


# ---------------------------------------------------------------------------
# 五、主流程
# ---------------------------------------------------------------------------


def render_report(
    lines: List[Tuple[str, float]],
    metrics: Optional[Dict[str, Any]] = None,
    reference: Optional[str] = None,
    engine_name: str = "",
    image_path: str = "",
) -> None:
    recognized = "\n".join(text for text, _ in lines)

    print("=" * 60)
    print(f"图片      : {image_path}")
    print(f"OCR 引擎  : {engine_name}")
    print("=" * 60)
    print("识别结果")
    print("-" * 60)
    if not lines:
        print("(未识别出任何文字)")
    for idx, (text, score) in enumerate(lines, 1):
        print(f"  {idx:>3}. {text}   [置信度 {score * 100:6.2f}%]")
    print("-" * 60)
    if lines:
        avg_conf = sum(s for _, s in lines) / len(lines)
        print(f"  平均置信度: {avg_conf * 100:.2f}%")
    print("识别文本:")
    print(recognized if recognized else "(空)")

    if metrics is None:
        print("=" * 60)
        print("提示: 提供标注文本 (--gt) 才能计算识别准确率。")
        return

    print("=" * 60)
    print("准确率评估")
    print("-" * 60)
    print(f"  标注字符数          : {metrics['ref_chars']}")
    print(f"  识别字符数          : {metrics['hyp_chars']}")
    print(f"  字符编辑距离        : {metrics['char_edit_distance']}")
    print(f"  字符准确率 (1-CER)  : {metrics['char_accuracy'] * 100:.2f}%"
          f"   (CER = {metrics['cer'] * 100:.2f}%)")
    print(f"  词数 (标注/识别)    : {metrics['ref_words']} / {metrics['hyp_words']}")
    print(f"  词准确率   (1-WER)  : {metrics['word_accuracy'] * 100:.2f}%"
          f"   (WER = {metrics['wer'] * 100:.2f}%)")
    print(f"  序列相似度          : {metrics['sequence_ratio'] * 100:.2f}%")
    print(f"  行级准确率          : {metrics['line_accuracy'] * 100:.2f}%"
          f"   ({metrics['line_hit']}/{metrics['line_total']} 行完全一致)")
    print(f"  完全匹配            : {'是' if metrics['exact_match'] else '否'}")
    print("  (行级准确率对换行切分方式敏感；字符/词准确率不受换行影响)")
    print("=" * 60)

    if reference is not None:
        ref_lines = split_lines(reference)
        hyp_lines = split_lines(recognized)
        diff: List[Tuple[int, str, str]] = []
        for i in range(max(len(ref_lines), len(hyp_lines))):
            ref_line = ref_lines[i] if i < len(ref_lines) else "(无此行)"
            hyp_line = hyp_lines[i] if i < len(hyp_lines) else "(无此行)"
            if ref_line != hyp_line:
                diff.append((i + 1, ref_line, hyp_line))
        if diff:
            print(f"存在差异的行 (共 {len(diff)} 行):")
            for no, ref_line, hyp_line in diff:
                print(f"  [第 {no} 行]")
                print(f"    标注: {ref_line}")
                print(f"    识别: {hyp_line}")
            print("=" * 60)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="图片文字识别 (OCR) 并评估识别准确率",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", nargs="?", help="待识别的图片路径")
    parser.add_argument("--gt", help="标注文本：直接给出字符串，或给出文本文件路径")
    parser.add_argument("--demo", action="store_true",
                        help="生成一张测试图片并评估识别准确率")
    parser.add_argument("--save", help="--demo 模式下测试图片的保存路径")
    parser.add_argument("--engine", default="auto",
                        choices=["auto", "rapidocr", "pytesseract"],
                        help="指定 OCR 引擎，默认自动选择")
    parser.add_argument("--ignore-punct", action="store_true",
                        help="计算准确率时忽略标点符号")
    parser.add_argument("--keep-case", action="store_true",
                        help="计算准确率时区分大小写")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出结果")
    parser.add_argument("--font-size", type=int, default=38,
                        help="--demo 生成图片时的字号")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="识别前把图片放大指定倍数（如 2），可提升小字识别率")
    parser.add_argument("--unclip", type=float, default=DEFAULT_UNCLIP_RATIO,
                        help="检测框扩张系数，越大越不容易裁掉行首/行尾字符"
                             "（默认 %.1f；设为 1.6 恢复引擎原行为）" % DEFAULT_UNCLIP_RATIO)
    args = parser.parse_args(argv)

    # 1. 准备图片与标注
    if args.demo:
        image_path = args.save or os.path.join(tempfile.gettempdir(), "ocr_demo.png")
        reference = "\n".join(make_demo_image(image_path, font_size=args.font_size))
        print(f"[demo] 已生成测试图片: {image_path}")
    else:
        if not args.image:
            parser.print_help()
            return 2
        image_path = args.image
        if not os.path.exists(image_path):
            print(f"错误: 找不到图片 {image_path}", file=sys.stderr)
            return 2
        reference = args.gt
        if reference and os.path.exists(reference) and os.path.isfile(reference):
            with open(reference, "r", encoding="utf-8") as fp:
                reference = fp.read()

    # 2. 识别
    tmp_path = None
    try:
        work_path, tmp_path = prepare_image(image_path, args.scale)
        if args.scale != 1.0:
            print(f"[scale] 图片已放大 {args.scale:g}x 后识别")
        engine = build_engine(args.engine, unclip_ratio=args.unclip)
        ocr_lines = engine.run_full(work_path)
        if args.scale != 1.0:
            # 检测框是在放大后的图上算出来的，换算回原图坐标
            ocr_lines = [
                ln._replace(box=[(x / args.scale, y / args.scale) for x, y in ln.box]
                            if ln.box else None)
                for ln in ocr_lines
            ]
        lines = [(ln.text, ln.score) for ln in ocr_lines]
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    recognized = "\n".join(text for text, _ in lines)

    # 3. 评估
    metrics = None
    if reference is not None:
        metrics = evaluate(
            reference,
            recognized,
            ignore_punct=args.ignore_punct,
            keep_case=args.keep_case,
        )

    if args.json:
        payload = {
            "image": image_path,
            "engine": engine.name,
            "unclip_ratio": args.unclip,
            "scale": args.scale,
            "lines": [
                {"text": ln.text, "confidence": ln.score,
                 "box": [[round(x, 1), round(y, 1)] for x, y in ln.box] if ln.box else None}
                for ln in ocr_lines
            ],
            "recognized_text": recognized,
            "average_confidence": (sum(s for _, s in lines) / len(lines)) if lines else 0.0,
            "metrics": metrics,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        render_report(lines, metrics, reference, engine.name, image_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
