import cv2, glob, os, numpy as np, math, argparse
def psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    return 20 * math.log10(255.0 / math.sqrt(mse)) if mse else 1e4

parser = argparse.ArgumentParser()
parser.add_argument('--ref_dir')   # ground-truth PNG 資料夾
parser.add_argument('--test_dir')  # 測試 PNG 資料夾
args = parser.parse_args()

ref_imgs = sorted(glob.glob(os.path.join(args.ref_dir, '*.png')))
test_imgs = sorted(glob.glob(os.path.join(args.test_dir, '*.png')))
psnrs = []
for r, t in zip(ref_imgs, test_imgs):
    ref = cv2.imread(r)
    test = cv2.imread(t)
    psnrs.append(psnr(ref, test))
print(f'Average PSNR: {np.mean(psnrs):.2f} dB')
