"""Local-only Earth2Ocean application; no additional web dependencies."""
import base64
import binascii
import colorsys
import io
import json
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from model_adapter import Earth2OceanSegmentation, infer, set_query_words, torch, np, Image
from PIL import ImageOps

TRANSLATIONS = {"鱼类":"fish", "鱼":"fish", "金枪鱼":"tuna", "鲨鱼":"shark",
    "海龟":"sea turtle", "章鱼":"octopus", "小丑鱼":"clownfish", "水母":"jellyfish",
    "海豚":"dolphin", "虾":"shrimp", "蟹":"crab", "珊瑚":"coral", "海草":"seagrass",
    "海床":"seabed", "水体背景":"underwater background", "背景":"underwater background"}
model = None
cache = {}
Image.MAX_IMAGE_PIXELS = 12_000_000

def png(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()

class Handler(BaseHTTPRequestHandler):
    def send(self, status, data, mime="application/json; charset=utf-8"):
        if isinstance(data, dict):
            data = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/live.js":
            return self.send(200, (ROOT / "live.js").read_bytes(), "text/javascript; charset=utf-8")
        if self.path in ("/", "/index.html"):
            return self.send(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        if self.path == "/health":
            return self.send(200, {"ready": True, "model": "Earth2Ocean ViT-B/16", "mllm": False})
        if self.path == "/sample.jpg":
            path = ROOT.parent / "Earth2Ocean-code/datasets/MAS3K/img/Turtle_541.jpg"
            if not path.is_file():
                return self.send(404, {"error":"本机未安装海龟示例照片，请上传自己的水下图片"})
            return self.send(200, path.read_bytes(), "image/jpeg")
        self.send(404, {"error":"未找到资源"})

    def do_POST(self):
        if self.path != "/segment":
            return self.send(404, {"error":"未找到接口"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 22_000_000:
                return self.send(413, {"error":"上传内容过大或为空"})
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("请求格式错误，请提交图片和类别")
            names = data.get("classes")
            if not isinstance(names, list) or not 1 <= len(names) <= 15:
                raise ValueError("请输入 1–15 个目标类别")
            words, labels = ["underwater background"], ["水体背景"]
            for name in names:
                if not isinstance(name, str) or not 0 < len(name.strip()) <= 80:
                    raise ValueError("类别名称须为 1–80 个字符")
                name = name.strip()
                word = TRANSLATIONS.get(name, name)
                if not word.isascii():
                    raise ValueError("暂不支持该中文类别，请填写英文名称：" + name)
                if word.lower() not in words:
                    words.append(word.lower())
                    labels.append(name)
            encoded_image = data.get("image")
            if not isinstance(encoded_image, str):
                raise ValueError("请提交有效的图片")
            try:
                raw = base64.b64decode(encoded_image.split(",", 1)[-1], validate=True)
            except binascii.Error:
                raise ValueError("图片数据损坏，请重新上传 JPG/PNG") from None
            if len(raw) > 15*1024*1024:
                raise ValueError("图片不能超过 15 MB")
            with Image.open(io.BytesIO(raw)) as source:
                if source.format not in ("JPEG", "PNG") or source.width * source.height > 12_000_000:
                    raise ValueError("请使用不超过 1200 万像素的 JPG/PNG")
                rgb = ImageOps.exif_transpose(source).convert("RGB")
            start = time.perf_counter()
            with torch.inference_mode():
                cache.clear()
                set_query_words(model, words, cache)
                ids = infer(model, rgb, "uploaded-image.jpg")
            colors = [(30, 55, 70)] + [tuple(int(v*255) for v in colorsys.hsv_to_rgb((i*.618+.09)%1, .62, 1)) for i in range(len(words)-1)]
            pixels = np.asarray(rgb).copy()
            fg = ids != 0
            pixels[fg] = (pixels[fg]*.55 + np.asarray(colors)[ids[fg]]*.45).astype(np.uint8)
            self.send(200, {"original":png(rgb), "overlay":png(Image.fromarray(pixels)),
                "mask":png(Image.fromarray(ids)), "seconds":round(time.perf_counter()-start, 2),
                "width":rgb.width,"height":rgb.height,
                "classes":[{"id":i,"name":label,"prompt":words[i],"color":"#%02x%02x%02x"%colors[i],
                            "area_percent":round(float((ids==i).mean()*100), 2)} for i,label in enumerate(labels)],
                "model":"Earth2Ocean ViT-B/16; underwater templates; no MLLM"})
        except (ValueError, KeyError, OSError, Image.DecompressionBombError) as exc:
            self.send(400, {"error":str(exc)})
        except Exception:
            traceback.print_exc()
            torch.cuda.empty_cache()
            self.send(500, {"error":"模型推理失败，请缩小图片或查看服务日志"})

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    classes_path = ROOT / "initial_classes.txt"
    model = Earth2OceanSegmentation(
        clip_type="pretrained_ckpt/vitb_laion2b_s34b_b88k.bin", model_type="ViT-B/16",
        json_path=None,imgname_list_path=None,name_path=str(classes_path),
        use_uwtemplate=True,use_mllm=False).to("cuda").eval()
    print("READY http://127.0.0.1:8765", flush=True)
    HTTPServer(("127.0.0.1",8765), Handler).serve_forever()
