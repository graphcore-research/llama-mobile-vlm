from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO = SCRIPT_DIR / "demo.mp4"
DEQUANT_ASM = SCRIPT_DIR / "dequant.s"
OUTPUT = SCRIPT_DIR / "headline.gif"

WIDTH = 1200
HEIGHT = 630
FPS = 10
CLIP_START_SECONDS = 19.5
CLIP_DURATION_SECONDS = 7

BACKGROUND = "#fffefa"
INK = "#25282b"
MUTED_INK = "#5d6368"
PINK = "#ff6d81"
PINK_DETAIL = "#e84f67"
TEAL = "#2cc4a7"
TEAL_DETAIL = "#159a84"
SHADOW = "#dedbd6"
PHONE = "#202326"

REGULAR_FONT = SCRIPT_DIR / "Graphik-Regular-Web.woff2"
MEDIUM_FONT = SCRIPT_DIR / "Graphik-Medium-Web.woff2"
DISPLAY_FONT = SCRIPT_DIR / "GraphcoreQuantized-Mixed.otf"
MONO_FONT = SCRIPT_DIR / "DejaVuSansMono.ttf"

missing_fonts = [
    path
    for path in (REGULAR_FONT, MEDIUM_FONT, DISPLAY_FONT, MONO_FONT)
    if not path.exists()
]
if missing_fonts:
    print(
        "Font files are not included in the repository; provide "
        f"{', '.join(path.name for path in missing_fonts)} in {SCRIPT_DIR}."
    )


with tempfile.TemporaryDirectory(prefix="llama-mobile-headline-") as tmp_name:
    tmp = Path(tmp_name)
    video_frames = tmp / "video-%03d.png"
    composite_frames = tmp / "frame-%03d.png"

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(CLIP_START_SECONDS),
            "-i",
            str(VIDEO),
            "-t",
            str(CLIP_DURATION_SECONDS),
            "-vf",
            f"fps={FPS}",
            str(video_frames),
        ],
        check=True,
    )

    canvas = Image.new("RGB", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(canvas)
    gradient_left = (255, 247, 248)
    gradient_right = (244, 255, 252)
    for x in range(WIDTH):
        amount = x / (WIDTH - 1)
        colour = tuple(
            round(left + (right - left) * amount)
            for left, right in zip(gradient_left, gradient_right)
        )
        draw.line((x, 0, x, HEIGHT), fill=colour)
    clean_background = canvas.copy()

    dequant_font = ImageFont.truetype(MONO_FONT, 12)
    dequant_lines = DEQUANT_ASM.read_text().splitlines()
    for code_x in (-45, 555):
        for code_y in (-55, 205, 465):
            for line_number, line in enumerate(dequant_lines):
                draw.text(
                    (code_x, code_y + line_number * 22),
                    line,
                    font=dequant_font,
                    fill="#d6dfdb",
                )

    original_box = (45, 290, 250, 440)
    qat_box = (335, 290, 540, 440)
    s3d8_box = (625, 290, 830, 440)
    phone_box = (942, 105, 1186, 625)
    screen_box = (954, 131, 1174, 608)

    protection_mask = Image.new("L", (WIDTH, HEIGHT), 0)
    protection_draw = ImageDraw.Draw(protection_mask)
    protection_draw.rounded_rectangle(
        (20, 10, 600, 182), radius=28, fill=255
    )
    for box in (original_box, qat_box, s3d8_box):
        protection_draw.rounded_rectangle(
            (box[0] - 14, box[1] - 14, box[2] + 14, box[3] + 14),
            radius=30,
            fill=255,
        )
    for arrow_start, arrow_end in ((240, 345), (530, 635), (820, 942)):
        protection_draw.line(
            (arrow_start, 365, arrow_end, 365), fill=255, width=48
        )
    protection_draw.rounded_rectangle(
        (922, 58, WIDTH, HEIGHT), radius=42, fill=255
    )
    protection_mask = protection_mask.filter(ImageFilter.GaussianBlur(22))
    canvas.paste(clean_background, (0, 0), protection_mask)

    headline_font = ImageFont.truetype(MEDIUM_FONT, 68)
    title_font = ImageFont.truetype(DISPLAY_FONT, 34)
    title_number_font = ImageFont.truetype(MEDIUM_FONT, 34)
    node_label_font = ImageFont.truetype(REGULAR_FONT, 18)
    node_title_font = ImageFont.truetype(MEDIUM_FONT, 34)
    demo_label_font = ImageFont.truetype(MEDIUM_FONT, 18)

    headline_x = 48
    draw.text((headline_x, 24), "21 GB", font=headline_font, fill=PINK_DETAIL)
    headline_x += draw.textlength("21 GB", font=headline_font) + 22
    draw.line((headline_x, 65, headline_x + 42, 65), fill=MUTED_INK, width=5)
    draw.polygon(
        (
            (headline_x + 52, 65),
            (headline_x + 40, 57),
            (headline_x + 40, 73),
        ),
        fill=MUTED_INK,
    )
    headline_x += 74
    draw.text((headline_x, 24), "3.7 GB", font=headline_font, fill=TEAL_DETAIL)
    title_x = 48
    draw.text((title_x, 115), "Fitting an ", font=title_font, fill=INK)
    title_x += draw.textlength("Fitting an ", font=title_font)
    draw.text((title_x, 115), "11B", font=title_number_font, fill=INK)
    title_x += draw.textlength("11B", font=title_number_font)
    draw.text((title_x, 115), " VLM on a phone", font=title_font, fill=INK)

    draw.rounded_rectangle(original_box, radius=16, fill=PINK)
    draw.rounded_rectangle(
        qat_box,
        radius=16,
        fill="#ffffff",
        outline="#cbc7c0",
        width=2,
    )
    draw.rounded_rectangle(
        s3d8_box,
        radius=16,
        fill=TEAL,
    )
    draw.text(
        (147.5, 337),
        "bfloat16",
        font=ImageFont.truetype(MEDIUM_FONT, 27),
        fill=INK,
        anchor="mm",
    )
    draw.text(
        (147.5, 392),
        "Original 11B VLM",
        font=node_label_font,
        fill=INK,
        anchor="mm",
    )
    draw.text(
        (437.5, 337),
        "QAT",
        font=node_title_font,
        fill=INK,
        anchor="mm",
    )
    draw.text(
        (437.5, 392),
        "With prompt sampling",
        font=node_label_font,
        fill=MUTED_INK,
        anchor="mm",
    )
    draw.text(
        (727.5, 337),
        "S3D8",
        font=node_title_font,
        fill=INK,
        anchor="mm",
    )
    draw.text(
        (727.5, 392),
        "2.7 bits per weight",
        font=node_label_font,
        fill=INK,
        anchor="mm",
    )

    for arrow_start, arrow_end in ((250, 335), (540, 625), (830, 932)):
        arrow_head_length = 11
        arrow_head_half_height = 7
        draw.line(
            (arrow_start, 365, arrow_end - arrow_head_length, 365),
            fill=MUTED_INK,
            width=4,
        )
        draw.polygon(
            (
                (arrow_end, 365),
                (arrow_end - arrow_head_length, 365 - arrow_head_half_height),
                (arrow_end - arrow_head_length, 365 + arrow_head_half_height),
            ),
            fill=MUTED_INK,
        )

    draw.text(
        (1064, 82),
        "Working demo",
        font=demo_label_font,
        fill=MUTED_INK,
        anchor="mm",
    )
    draw.rounded_rectangle(
        (phone_box[0] + 5, phone_box[1] + 5, phone_box[2] + 5, phone_box[3] + 5),
        radius=30,
        fill=SHADOW,
    )
    draw.rounded_rectangle(phone_box, radius=30, fill=PHONE)
    draw.rounded_rectangle(screen_box, radius=13, fill="#ffffff")
    draw.rounded_rectangle((1039, 113, 1079, 118), radius=3, fill="#505459")
    draw.ellipse((1088, 111, 1095, 118), fill="#505459")

    screen_width = screen_box[2] - screen_box[0]
    screen_height = screen_box[3] - screen_box[1]
    screen_mask = Image.new("L", (screen_width, screen_height), 0)
    ImageDraw.Draw(screen_mask).rounded_rectangle(
        (0, 0, screen_width - 1, screen_height - 1), radius=12, fill=255
    )

    paths = sorted(tmp.glob("video-*.png"))
    for index, frame_path in enumerate(paths, start=1):
        with Image.open(frame_path) as video_frame:
            video_frame = video_frame.convert("RGB").crop((0, 80, 1080, 2380)).resize(
                (screen_width, screen_height), Image.Resampling.LANCZOS
            )
            frame = canvas.copy()
            frame.paste(video_frame, screen_box[:2], screen_mask)
            frame.save(tmp / f"frame-{index:03d}.png")

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-i",
            str(composite_frames),
            "-filter_complex",
            (
                "split[frames][palette_input];"
                "[palette_input]palettegen=max_colors=128:stats_mode=diff[palette];"
                "[frames][palette]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle"
            ),
            "-loop",
            "0",
            "-y",
            str(OUTPUT),
        ],
        check=True,
    )

print(OUTPUT)
