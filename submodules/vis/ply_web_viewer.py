#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""用网页可视化 3DGS 导出的 PLY 点云（常见为 binary_little_endian 1.0）。

特点：
- 纯标准库起服务（无需额外依赖）
- 前端用 Three.js + OrbitControls
- 支持从 3DGS 的 `f_dc_0/1/2` 近似还原颜色（SH DC 项：rgb = clamp(C0 * f_dc + 0.5)）

用法：
  python submodules/vis/ply_web_viewer.py --ply /abs/path/to/point_cloud.ply
"""

from __future__ import annotations

import argparse
import os
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>PLY Viewer</title>
    <style>
      html, body { height: 100%; margin: 0; background: #0b0f14; overflow: hidden; }
      #app { position: fixed; inset: 0; }
      .hud {
        position: fixed; top: 12px; left: 12px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 12px; line-height: 1.4;
        padding: 10px 12px; border-radius: 10px;
        background: rgba(20, 26, 34, 0.75); color: #e6edf3;
        backdrop-filter: blur(8px);
        user-select: none;
        max-width: min(520px, calc(100vw - 24px));
      }
      .hud b { color: #fff; }
      .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
      input[type="range"] { width: 220px; }
      .hint { opacity: 0.85; }
      .err { color: #ffb4b4; white-space: pre-wrap; }
      a { color: #8cc8ff; }
    </style>
    <script type="importmap">
      {
        "imports": {
          "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
          "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
        }
      }
    </script>
  </head>
  <body>
    <div id="app"></div>
    <div class="hud">
      <div class="row">
        <b>PLY Web Viewer</b>
        <span id="status" class="hint">加载中…</span>
      </div>
      <div id="meta" class="hint"></div>
      <div class="row" style="margin-top: 8px;">
        <span>点大小</span>
        <input id="size" type="range" min="1" max="30" step="1" value="6" />
        <span id="sizev">6</span>
      </div>
      <div class="row">
        <label><input id="axes" type="checkbox" checked /> 坐标轴</label>
        <label><input id="bg" type="checkbox" /> 亮背景</label>
      </div>
      <div class="hint">鼠标左键旋转 / 右键平移 / 滚轮缩放</div>
      <div id="error" class="err"></div>
    </div>

    <script type="module">
      import * as THREE from 'three';
      import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

      const DATA_URL = '/data.ply';
      const C0 = 0.28209479177387814;

      const elStatus = document.getElementById('status');
      const elMeta = document.getElementById('meta');
      const elErr = document.getElementById('error');
      const elSize = document.getElementById('size');
      const elSizeV = document.getElementById('sizev');
      const elAxes = document.getElementById('axes');
      const elBg = document.getElementById('bg');

      function clamp01(x) { return Math.min(1, Math.max(0, x)); }

      function findEndHeader(text) {
        // end_header 行本身也要包含
        const idx = text.indexOf('end_header\n');
        if (idx !== -1) return idx + 'end_header\n'.length;
        const idx2 = text.indexOf('end_header\r\n');
        if (idx2 !== -1) return idx2 + 'end_header\r\n'.length;
        return -1;
      }

      function parseHeader(headerText) {
        const lines = headerText.split(/\r?\n/).filter(Boolean);
        let format = null;
        let vertexCount = null;
        let inVertex = false;
        const vertexProps = [];

        for (const line of lines) {
          const parts = line.trim().split(/\s+/);
          if (!parts.length) continue;
          if (parts[0] === 'format') {
            format = parts[1];
          } else if (parts[0] === 'element') {
            inVertex = parts[1] === 'vertex';
            if (inVertex) vertexCount = Number(parts[2]);
          } else if (parts[0] === 'property' && inVertex) {
            // 忽略 list property（面片等），这里只关心 vertex 的标量属性
            if (parts[1] === 'list') continue;
            const type = parts[1];
            const name = parts[2];
            vertexProps.push({ type, name });
          }
        }

        if (!format || !vertexCount || vertexCount <= 0) {
          throw new Error(`PLY header 解析失败：format=${format}, vertexCount=${vertexCount}`);
        }
        return { format, vertexCount, vertexProps };
      }

      function typeInfo(t) {
        // https://en.wikipedia.org/wiki/PLY_(file_format)
        // 常见类型：float/double/uchar/int 等。
        const map = {
          'char': { size: 1, get: (dv, o, le) => dv.getInt8(o) },
          'int8': { size: 1, get: (dv, o, le) => dv.getInt8(o) },
          'uchar': { size: 1, get: (dv, o, le) => dv.getUint8(o) },
          'uint8': { size: 1, get: (dv, o, le) => dv.getUint8(o) },
          'short': { size: 2, get: (dv, o, le) => dv.getInt16(o, le) },
          'int16': { size: 2, get: (dv, o, le) => dv.getInt16(o, le) },
          'ushort': { size: 2, get: (dv, o, le) => dv.getUint16(o, le) },
          'uint16': { size: 2, get: (dv, o, le) => dv.getUint16(o, le) },
          'int': { size: 4, get: (dv, o, le) => dv.getInt32(o, le) },
          'int32': { size: 4, get: (dv, o, le) => dv.getInt32(o, le) },
          'uint': { size: 4, get: (dv, o, le) => dv.getUint32(o, le) },
          'uint32': { size: 4, get: (dv, o, le) => dv.getUint32(o, le) },
          'float': { size: 4, get: (dv, o, le) => dv.getFloat32(o, le) },
          'float32': { size: 4, get: (dv, o, le) => dv.getFloat32(o, le) },
          'double': { size: 8, get: (dv, o, le) => dv.getFloat64(o, le) },
          'float64': { size: 8, get: (dv, o, le) => dv.getFloat64(o, le) },
        };
        const info = map[t];
        if (!info) throw new Error(`不支持的 PLY property type: ${t}`);
        return info;
      }

      function parseBinaryLittleEndian(buffer, headerBytes, header) {
        const { vertexCount, vertexProps } = header;
        const dv = new DataView(buffer, headerBytes);
        const le = true;

        // 自动下采样，避免超大点云把浏览器内存打爆
        const MAX_POINTS = 2_000_000;
        const step = (vertexCount > MAX_POINTS) ? Math.ceil(vertexCount / MAX_POINTS) : 1;
        const outCount = Math.ceil(vertexCount / step);

        // 预计算每个 property 的 offset，后面只读取必要字段
        const infos = [];
        let stride = 0;
        for (const p of vertexProps) {
          const ti = typeInfo(p.type);
          infos.push({ ...p, ...ti, off: stride });
          stride += ti.size;
        }
        const need = stride * vertexCount;
        if (dv.byteLength < need) {
          throw new Error(`PLY 数据区长度不足：need=${need}, got=${dv.byteLength}`);
        }

        const find = (name) => infos.find(p => p.name === name);
        const px = find('x'), py = find('y'), pz = find('z');
        if (!px || !py || !pz) throw new Error('PLY vertex 缺少 x/y/z');

        const pr = find('red'), pg = find('green'), pb = find('blue');
        const pdc0 = find('f_dc_0'), pdc1 = find('f_dc_1'), pdc2 = find('f_dc_2');

        const pos = new Float32Array(outCount * 3);
        const col = new Float32Array(outCount * 3);

        let colorMode = 'default';
        if (pr && pg && pb) colorMode = 'rgb';
        else if (pdc0 && pdc1 && pdc2) colorMode = 'f_dc';

        let out = 0;
        for (let i = 0; i < vertexCount; i += step) {
          const base = i * stride;

          const x = px.get(dv, base + px.off, le);
          const y = py.get(dv, base + py.off, le);
          const z = pz.get(dv, base + pz.off, le);
          pos[out * 3 + 0] = x;
          pos[out * 3 + 1] = y;
          pos[out * 3 + 2] = z;

          if (colorMode === 'rgb') {
            const r = pr.get(dv, base + pr.off, le);
            const g = pg.get(dv, base + pg.off, le);
            const b = pb.get(dv, base + pb.off, le);
            // 有的 PLY 是 uchar(0..255)，也可能是 float(0..1)
            const rf = (r > 1.0) ? (r / 255.0) : r;
            const gf = (g > 1.0) ? (g / 255.0) : g;
            const bf = (b > 1.0) ? (b / 255.0) : b;
            col[out * 3 + 0] = clamp01(rf);
            col[out * 3 + 1] = clamp01(gf);
            col[out * 3 + 2] = clamp01(bf);
          } else if (colorMode === 'f_dc') {
            const dc0 = pdc0.get(dv, base + pdc0.off, le);
            const dc1 = pdc1.get(dv, base + pdc1.off, le);
            const dc2 = pdc2.get(dv, base + pdc2.off, le);
            // 3DGS 常见导出：SH2RGB = clamp(C0 * f_dc + 0.5)
            col[out * 3 + 0] = clamp01(C0 * dc0 + 0.5);
            col[out * 3 + 1] = clamp01(C0 * dc1 + 0.5);
            col[out * 3 + 2] = clamp01(C0 * dc2 + 0.5);
          } else {
            col[out * 3 + 0] = 1.0;
            col[out * 3 + 1] = 1.0;
            col[out * 3 + 2] = 1.0;
          }

          out++;
        }

        return { pos, col, colorMode, sampledFrom: vertexCount, sampledTo: outCount, step };
      }

      async function loadPlyGeometry(url) {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`下载失败: ${res.status} ${res.statusText}`);
        const buffer = await res.arrayBuffer();

        // 只解码前 256KB 以定位 header
        const headLen = Math.min(buffer.byteLength, 256 * 1024);
        const headText = new TextDecoder('utf-8').decode(new Uint8Array(buffer, 0, headLen));
        const endIdx = findEndHeader(headText);
        if (endIdx < 0) throw new Error('找不到 PLY end_header');

        const headerText = headText.slice(0, endIdx);
        const header = parseHeader(headerText);

        const headerBytes = new TextEncoder().encode(headerText).byteLength;
        if (header.format === 'binary_little_endian') {
          const { pos, col, colorMode, sampledFrom, sampledTo, step } = parseBinaryLittleEndian(buffer, headerBytes, header);
          const geometry = new THREE.BufferGeometry();
          geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3));
          geometry.setAttribute('color', new THREE.BufferAttribute(col, 3));
          geometry.computeBoundingSphere();
          geometry.computeBoundingBox();
          return { geometry, meta: { ...header, colorMode, sampledFrom, sampledTo, step } };
        }

        if (header.format === 'ascii') {
          // 简化支持：只适用于小文件。3DGS 默认导出通常为 binary_little_endian。
          const text = new TextDecoder('utf-8').decode(new Uint8Array(buffer));
          const fullEnd = findEndHeader(text);
          const body = text.slice(fullEnd).trim().split(/\r?\n/);
          const { vertexCount, vertexProps } = header;
          const nameToIdx = new Map(vertexProps.map((p, i) => [p.name, i]));
          const ix = nameToIdx.get('x'), iy = nameToIdx.get('y'), iz = nameToIdx.get('z');
          if (ix === undefined || iy === undefined || iz === undefined) throw new Error('PLY vertex 缺少 x/y/z');
          const ir = nameToIdx.get('red'), ig = nameToIdx.get('green'), ib = nameToIdx.get('blue');
          const idc0 = nameToIdx.get('f_dc_0'), idc1 = nameToIdx.get('f_dc_1'), idc2 = nameToIdx.get('f_dc_2');

          const pos = new Float32Array(vertexCount * 3);
          const col = new Float32Array(vertexCount * 3);
          let colorMode = 'default';
          if (ir !== undefined && ig !== undefined && ib !== undefined) colorMode = 'rgb';
          else if (idc0 !== undefined && idc1 !== undefined && idc2 !== undefined) colorMode = 'f_dc';

          for (let i = 0; i < vertexCount; i++) {
            const parts = body[i].trim().split(/\s+/);
            const x = Number(parts[ix]), y = Number(parts[iy]), z = Number(parts[iz]);
            pos[i * 3 + 0] = x;
            pos[i * 3 + 1] = y;
            pos[i * 3 + 2] = z;
            if (colorMode === 'rgb') {
              const r = Number(parts[ir]), g = Number(parts[ig]), b = Number(parts[ib]);
              col[i * 3 + 0] = clamp01((r > 1.0) ? (r / 255.0) : r);
              col[i * 3 + 1] = clamp01((g > 1.0) ? (g / 255.0) : g);
              col[i * 3 + 2] = clamp01((b > 1.0) ? (b / 255.0) : b);
            } else if (colorMode === 'f_dc') {
              const dc0 = Number(parts[idc0]), dc1 = Number(parts[idc1]), dc2 = Number(parts[idc2]);
              col[i * 3 + 0] = clamp01(C0 * dc0 + 0.5);
              col[i * 3 + 1] = clamp01(C0 * dc1 + 0.5);
              col[i * 3 + 2] = clamp01(C0 * dc2 + 0.5);
            } else {
              col[i * 3 + 0] = 1;
              col[i * 3 + 1] = 1;
              col[i * 3 + 2] = 1;
            }
          }

          const geometry = new THREE.BufferGeometry();
          geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3));
          geometry.setAttribute('color', new THREE.BufferAttribute(col, 3));
          geometry.computeBoundingSphere();
          geometry.computeBoundingBox();
          return { geometry, meta: { ...header, colorMode } };
        }

        throw new Error(`暂不支持的 PLY format: ${header.format}`);
      }

      function setError(msg) {
        elErr.textContent = String(msg || '');
      }

      const container = document.getElementById('app');
      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setSize(window.innerWidth, window.innerHeight);
      container.appendChild(renderer.domElement);

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0b0f14);

      const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.001, 1e6);
      camera.position.set(0, 0, 3);

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;

      const light = new THREE.DirectionalLight(0xffffff, 0.8);
      light.position.set(1, 1, 1);
      scene.add(light);
      scene.add(new THREE.AmbientLight(0xffffff, 0.3));

      const axes = new THREE.AxesHelper(1);
      scene.add(axes);

      let points = null;
      let material = null;
      let bbox = null;

      function updatePointSize() {
        if (!material || !bbox) return;
        const v = Number(elSize.value);
        elSizeV.textContent = String(v);
        // 根据包围盒对点大小做一个自适应缩放
        const diag = bbox.max.clone().sub(bbox.min).length();
        const base = (diag > 0) ? (diag / 700.0) : 0.01;
        material.size = base * (v / 6.0);
        material.needsUpdate = true;
      }

      function resize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      }
      window.addEventListener('resize', resize);

      elAxes.addEventListener('change', () => { axes.visible = elAxes.checked; });
      elBg.addEventListener('change', () => {
        scene.background = new THREE.Color(elBg.checked ? 0xf5f7fb : 0x0b0f14);
      });
      elSize.addEventListener('input', updatePointSize);

      async function main() {
        try {
          elStatus.textContent = '下载/解析 PLY…';
          const { geometry, meta } = await loadPlyGeometry(DATA_URL);
          bbox = geometry.boundingBox;

          // 将点云中心移动到原点
          const center = bbox.getCenter(new THREE.Vector3());
          geometry.translate(-center.x, -center.y, -center.z);
          geometry.computeBoundingSphere();
          geometry.computeBoundingBox();
          bbox = geometry.boundingBox;

          material = new THREE.PointsMaterial({
            size: 0.01,
            sizeAttenuation: true,
            vertexColors: true,
            transparent: false,
          });
          updatePointSize();

          points = new THREE.Points(geometry, material);
          scene.add(points);

          // 相机自动对准
          const sphere = geometry.boundingSphere;
          const r = (sphere && sphere.radius) ? sphere.radius : 1;
          camera.position.set(0, 0, Math.max(0.1, r * 2.5));
          controls.target.set(0, 0, 0);
          controls.update();

          elStatus.textContent = '完成';
          const pShow = (meta.sampledTo && meta.sampledFrom && meta.step && meta.step > 1)
            ? `${meta.sampledTo.toLocaleString()} (from ${meta.sampledFrom.toLocaleString()}, step=${meta.step})`
            : meta.vertexCount.toLocaleString();
          elMeta.textContent = `points=${pShow} | format=${meta.format} | color=${meta.colorMode}`;
          setError('');
        } catch (e) {
          elStatus.textContent = '失败';
          setError(e && e.stack ? e.stack : e);
        }
      }

      function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      }

      main();
      animate();
    </script>
  </body>
</html>
"""


class _PLYViewerHandler(BaseHTTPRequestHandler):
    def do_HEAD(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            body = HTML_TEMPLATE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return

        if path == "/data.ply":
            ply_path = getattr(self.server, "ply_path", None)
            if not ply_path or not os.path.isfile(ply_path):
                self.send_error(HTTPStatus.NOT_FOUND, "PLY 文件不存在")
                return
            fs = os.stat(ply_path)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(fs.st_size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return

        if path == "/health":
            body = b"ok"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            body = HTML_TEMPLATE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/data.ply":
            ply_path = getattr(self.server, "ply_path", None)
            if not ply_path or not os.path.isfile(ply_path):
                self.send_error(HTTPStatus.NOT_FOUND, "PLY 文件不存在")
                return

            try:
                fs = os.stat(ply_path)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(fs.st_size))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                with open(ply_path, "rb") as f:
                    while True:
                        chunk = f.read(1024 * 1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            except BrokenPipeError:
                return
            except Exception as e:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"读取 PLY 失败: {e}")
            return

        if path == "/health":
            body = b"ok"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        # 安静一点：默认 http.server 会把每个请求都打印出来
        return


def visualize_ply_web(
    ply_path: str,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = False,
) -> tuple[str, ThreadingHTTPServer]:
    """启动一个本地 HTTP 服务，用浏览器可视化 PLY。

    返回 (url, server)。server 为 daemon 线程运行；需要手动停止可调用 server.shutdown().
    """

    ply_path = os.path.abspath(os.path.expanduser(ply_path))
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(ply_path)

    server = ThreadingHTTPServer((host, port), _PLYViewerHandler)
    server.ply_path = ply_path  # type: ignore[attr-defined]

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    url = f"http://{host}:{server.server_port}/"
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    return url, server


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ply",
        required=True,
        help="要可视化的 PLY 文件路径（例如 3DGS 的 point_cloud.ply）",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="默认 0 表示自动选择空闲端口")
    ap.add_argument("--open", action="store_true", help="尝试自动打开浏览器")
    args = ap.parse_args()

    url, server = visualize_ply_web(args.ply, host=args.host, port=args.port, open_browser=args.open)
    print(f"[PLY Viewer] serving: {os.path.abspath(args.ply)}")
    print(f"[PLY Viewer] url: {url}")
    print("[PLY Viewer] 按 Ctrl+C 退出")
    try:
        while True:
            threading.Event().wait(3600)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    _main()
