/* Minimal libjpeg(-turbo) helpers used through ctypes (see bitstream.py).
 *
 * read_coefficients: entropy-decodes a baseline JPEG and returns the quantized DCT coefficients
 *   without inverse DCT, upsampling or colour conversion (jpeg_read_coefficients).
 * decode_pixels: full decode with the same library (inverse DCT, upsampling, colour conversion),
 *   so both paths are timed with identical code.
 *
 * Coefficients and quantization tables are returned in natural (row-major) order: index 8*u + v,
 * u = vertical and v = horizontal frequency, the same layout as revision/common.py:block_dct.
 * Build: bash revision/build_jpegcoef.sh
 */
#include <setjmp.h>
#include <stdio.h>
#include <string.h>
#include <jpeglib.h>

struct err_mgr {
    struct jpeg_error_mgr pub;
    jmp_buf jb;
};

static void on_error(j_common_ptr cinfo) {
    struct err_mgr *e = (struct err_mgr *)cinfo->err;
    longjmp(e->jb, 1);
}

/* info (int[4 * 4]): per component width_in_blocks, height_in_blocks, h_samp_factor, v_samp_factor.
 * out (short[out_size]): components one after another, each height_in_blocks x width_in_blocks x 64.
 * q (unsigned short[4 * 64]): quantization table of each component.
 * Returns the number of components, -1 on a decoding error, -2 if out is too small. */
int read_coefficients(const unsigned char *buf, unsigned long len, short *out, long out_size,
                      int *info, unsigned short *q) {
    struct jpeg_decompress_struct cinfo;
    struct err_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = on_error;
    if (setjmp(jerr.jb)) {
        jpeg_destroy_decompress(&cinfo);
        return -1;
    }
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, buf, len);
    jpeg_read_header(&cinfo, TRUE);
    jvirt_barray_ptr *coefs = jpeg_read_coefficients(&cinfo);
    int nc = cinfo.num_components;
    long pos = 0;
    for (int c = 0; c < nc && c < 4; c++) {
        jpeg_component_info *ci = &cinfo.comp_info[c];
        int wb = (int)ci->width_in_blocks, hb = (int)ci->height_in_blocks;
        info[4 * c] = wb;
        info[4 * c + 1] = hb;
        info[4 * c + 2] = ci->h_samp_factor;
        info[4 * c + 3] = ci->v_samp_factor;
        for (int k = 0; k < 64; k++)
            q[64 * c + k] = ci->quant_table ? ci->quant_table->quantval[k] : 0;
        if (pos + (long)wb * hb * 64 > out_size) {
            jpeg_destroy_decompress(&cinfo);
            return -2;
        }
        for (int by = 0; by < hb; by++) {
            JBLOCKARRAY rows = (*cinfo.mem->access_virt_barray)((j_common_ptr)&cinfo, coefs[c],
                                                                (JDIMENSION)by, 1, FALSE);
            memcpy(out + pos, rows[0], sizeof(JCOEF) * 64 * (size_t)wb);
            pos += (long)wb * 64;
        }
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return nc;
}

/* Full decode to interleaved 8-bit samples. color: 0 = RGB, 1 = YCbCr (no colour conversion).
 * wh (int[2]) receives width and height. Returns 0, -1 on a decoding error, -2 if out is too small. */
int decode_pixels(const unsigned char *buf, unsigned long len, unsigned char *out, long out_size,
                  int color, int *wh) {
    struct jpeg_decompress_struct cinfo;
    struct err_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = on_error;
    if (setjmp(jerr.jb)) {
        jpeg_destroy_decompress(&cinfo);
        return -1;
    }
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, buf, len);
    jpeg_read_header(&cinfo, TRUE);
    cinfo.out_color_space = color ? JCS_YCbCr : JCS_RGB;
    jpeg_start_decompress(&cinfo);
    long w = cinfo.output_width, h = cinfo.output_height, stride = w * cinfo.output_components;
    if (stride * h > out_size) {
        jpeg_destroy_decompress(&cinfo);
        return -2;
    }
    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW row = out + (long)cinfo.output_scanline * stride;
        jpeg_read_scanlines(&cinfo, &row, 1);
    }
    wh[0] = (int)w;
    wh[1] = (int)h;
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return 0;
}
