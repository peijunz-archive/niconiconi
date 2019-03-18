#!/usr/bin/env python3
import numpy as np
import cv2
import argparse

template = {
    "src": "nico.png",
    "anchor": [268, 25],
    "size": [430, 312]
}


def nico_book(template, bookname, outname):
    img_tpl_full = cv2.imread(template["src"], cv2.IMREAD_UNCHANGED)
    img_tpl = img_tpl_full[:, :, :3]
    alpha = img_tpl_full[:, :, 3:]/255.

    img_book = cv2.imread(bookname, cv2.IMREAD_COLOR)
    (x, y), (dx, dy) = template["anchor"], template["size"]
    s = slice(x, x+dx), slice(y, y+dy)
    img_book = cv2.resize(img_book, (dy, dx))
    img_tpl[s] = img_tpl[s]*alpha[s] + img_book*(1-alpha[s])

    if not outname:
        outname = "output-" + bookname
    cv2.imwrite(outname, img_tpl)


parser = argparse.ArgumentParser(
    description='Generate nico image with your favorite book')
parser.add_argument('book', metavar='book', type=str, nargs=1,
                    help='cover image. Optimal aspect ratio is ~1.35')
parser.add_argument('out', metavar='out', nargs='?',
                    help='output image with embeded book')

args = parser.parse_args()
nico_book(template, args.book[0], args.out)
