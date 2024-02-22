import numpy as np
import os

from utils.metric_data_rgbd import test_dataset
from utils.saliency_metric import cal_mae,cal_fm,cal_sm,cal_em,cal_wfm


gt_path = '../datasets/rgbd/test/'
prediction_path = '../predictions/unitr_rgbd_swin/'
# prediction_path = '../predictions/unitr_rgbd_res/'
datasets = ['SIP','NJU2K','NLPR','STERE']

for i in datasets:
    sal_root = os.path.join(prediction_path, i + '/')
    gt_root = gt_path + i + '/GT/'
    test_loader = test_dataset(sal_root, gt_root)
    mae, fm, sm, em, wfm = cal_mae(), cal_fm(test_loader.size), cal_sm(), cal_em(), cal_wfm()
    for j in range(test_loader.size):
        sal, gt = test_loader.load_data(s=i)
        if sal.size != gt.size:
            x, y = gt.size
            sal = sal.resize((x, y))
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        gt[gt > 0.5] = 1
        gt[gt != 1] = 0
        res = sal
        res = np.array(res)
        if res.max() == res.min():
            res = res/255
        else:
            res = (res - res.min()) / (res.max() - res.min())
        mae.update(res, gt)
        sm.update(res, gt)
        fm.update(res, gt)
        em.update(res, gt)
        wfm.update(res, gt)

    MAE = mae.show()
    maxf, meanf, _, _ = fm.show()
    sm = sm.show()
    em = em.show()
    wfm = wfm.show()

    print('dataset: {} Em: {:.3f}   Sm: {:.3f}   maxF: {:.3f}   wfm: {:.3f}   avgF: {:.3f}   MAE: {:.3f}'.format(
        i, em, sm, maxf, wfm, meanf, MAE))
