"""Read DCT coefficients directly from JPEG files (no inverse DCT, no colour conversion) via libjpegcoef.so.

coefficient_tensor() returns the dequantized coefficients of a 4:4:4 JPEG as a (64*C, H/8, W/8) float32 array
with channel 64*c + 8*u + v, the same layout as common.block_dct applied to JPEG-convention YCbCr
(full-range BT.601 on 0-255 RGB, level shift -128). A DCT-domain model trained on that input can therefore
classify the bitstream coefficients directly.
  python revision/bitstream.py --selftest
"""
import argparse
import ctypes
import io
import os

import numpy as np

_LIB = None
_BUF = {"coef": np.zeros(3 * 128 * 128 * 64, np.int16), "pix": np.zeros(1024 * 1024 * 3, np.uint8)}


def lib():
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libjpegcoef.so"))
        _LIB.read_coefficients.argtypes = [ctypes.c_char_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_short),
                                           ctypes.c_long, ctypes.POINTER(ctypes.c_int),
                                           ctypes.POINTER(ctypes.c_ushort)]
        _LIB.decode_pixels.argtypes = [ctypes.c_char_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ubyte),
                                       ctypes.c_long, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    return _LIB


def read_coefficients(data):
    """Quantized coefficients: list of int16 arrays (H_blocks, W_blocks, 64) per component, and the
    quantization tables (components x 64), both in natural order."""
    info = np.zeros(16, np.int32)
    q = np.zeros(4 * 64, np.uint16)
    while True:
        buf = _BUF["coef"]
        nc = lib().read_coefficients(data, len(data), buf.ctypes.data_as(ctypes.POINTER(ctypes.c_short)), buf.size,
                                     info.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                     q.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort)))
        if nc == -2:
            _BUF["coef"] = np.zeros(buf.size * 4, np.int16)
            continue
        if nc < 0:
            raise ValueError("JPEG decoding error")
        break
    comps, pos = [], 0
    for c in range(nc):
        wb, hb = int(info[4 * c]), int(info[4 * c + 1])
        comps.append(buf[pos:pos + hb * wb * 64].reshape(hb, wb, 64).copy())
        pos += hb * wb * 64
    return comps, q[:nc * 64].reshape(nc, 64).astype(np.float32)


def coefficient_tensor(data):
    """Dequantized coefficients of a 4:4:4 JPEG as float32 (64 * components, H/8, W/8)."""
    comps, q = read_coefficients(data)
    if any(c.shape != comps[0].shape for c in comps):
        raise ValueError("chroma is subsampled; coefficient_tensor expects 4:4:4 JPEG files")
    x = np.stack(comps).astype(np.float32) * q[:, None, None, :]        # C, Hb, Wb, 64
    c, hb, wb, _ = x.shape
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2).reshape(c * 64, hb, wb))


def decode(data, ycbcr=False):
    """Full libjpeg decode to uint8 (H, W, 3); ycbcr=True skips the colour conversion."""
    wh = np.zeros(2, np.int32)
    while True:
        buf = _BUF["pix"]
        r = lib().decode_pixels(data, len(data), buf.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), buf.size,
                                int(ycbcr), wh.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        if r == -2:
            _BUF["pix"] = np.zeros(buf.size * 4, np.uint8)
            continue
        if r < 0:
            raise ValueError("JPEG decoding error")
        w, h = int(wh[0]), int(wh[1])
        return buf[:h * w * 3].reshape(h, w, 3).copy()


def dct_matrix():
    n = np.arange(8)
    t = np.cos(np.pi * (2 * n[None, :] + 1) * n[:, None] / 16) * np.sqrt(2 / 8)
    t[0] /= np.sqrt(2)
    return t


def block_dct_numpy(planes):
    """(C, H, W) float -> (64*C, H/8, W/8), channel 64*c + 8*u + v (same as common.block_dct)."""
    c, h, w = planes.shape
    t = dct_matrix()
    blocks = planes.reshape(c, h // 8, 8, w // 8, 8).transpose(0, 1, 3, 2, 4)      # C, Hb, Wb, 8, 8
    coef = t @ blocks @ t.T
    return coef.reshape(c, h // 8, w // 8, 64).transpose(0, 3, 1, 2).reshape(c * 64, h // 8, w // 8)


def selftest():
    from PIL import Image
    rng = np.random.RandomState(0)
    yy, xx = np.mgrid[0:224, 0:224]
    img = np.stack([128 + 60 * np.sin(xx / 9.0), 128 + 60 * np.cos(yy / 7.0), 128 + 40 * np.sin((xx + yy) / 13.0)], -1)
    img = np.clip(img + rng.randn(224, 224, 3) * 8, 0, 255).astype(np.uint8)
    for q, sub in [(95, 0), (50, 0)]:
        b = io.BytesIO()
        Image.fromarray(img).save(b, "JPEG", quality=q, subsampling=sub)
        data = b.getvalue()
        coef = coefficient_tensor(data)
        ycc = decode(data, ycbcr=True).astype(np.float64).transpose(2, 0, 1) - 128.0
        ref = block_dct_numpy(ycc)
        rgb = decode(data)
        pil = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        rel = np.abs(coef - ref).mean() / np.abs(ref).mean()
        print(f"Q={q} 4:4:4: coefficients {coef.shape}; mean |read - DCT(decoded)| / mean |DCT| = {rel:.4f}; "
              f"max |libjpeg RGB - Pillow RGB| = {np.abs(rgb.astype(int) - pil.astype(int)).max()}")
        assert coef.shape == (192, 28, 28) and rel < 0.05, "coefficients do not match the decoded image"
    b = io.BytesIO()
    Image.fromarray(img).save(b, "JPEG", quality=90, subsampling=2)
    comps, _ = read_coefficients(b.getvalue())
    print("4:2:0 component block grids:", [c.shape[:2] for c in comps])
    print("bitstream self-test passed")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    if p.parse_args().selftest:
        selftest()
