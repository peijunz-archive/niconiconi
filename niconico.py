#!/usr/bin/env python3
import cv2
import argparse

template = {
    "src_bg": "orig_bg.png",
    "src_fg": "orig_fg.png",
    "anchor": [268, 25],
    "size": [430, 312]
}


def nico_book(template, bookname, outname):
    img_tpl_full = cv2.imread(template["src_bg"], cv2.IMREAD_UNCHANGED)
    img_tpl = img_tpl_full[:, :, :3]
    alpha = img_tpl_full[:, :, 3:]/255.

    img_book = cv2.imread(bookname, cv2.IMREAD_COLOR)
    (x, y), (dx, dy) = template["anchor"], template["size"]
    s = slice(x, x+dx), slice(y, y+dy)
    img_book = cv2.resize(img_book, (dy, dx))
    img_tpl[s] = img_tpl[s]*alpha[s] + img_book*(1-alpha[s])


    img_tpl_fg = cv2.imread(template["src_fg"], cv2.IMREAD_UNCHANGED)
    img_fg = img_tpl_fg[:, :, :3]
    alpha = img_tpl_fg[:, :, 3:]/255.

    img_tpl = img_tpl*(1-alpha) + img_fg*(alpha)

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
