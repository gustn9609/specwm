import cv2, sys
p, t = sys.argv[1], int(sys.argv[2]); c = cv2.VideoCapture(p); c.set(cv2.CAP_PROP_POS_FRAMES, t); ok, f = c.read(); cv2.imwrite(sys.argv[3], f, [cv2.IMWRITE_JPEG_QUALITY, 88]); print(ok, f.shape if ok else None)
