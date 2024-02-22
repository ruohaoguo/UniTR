import datetime
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from val import validation
from loss import Loss, Video_Loss
from utils.tools import custom_print
from utils.data_processed import VideoDataset

writer = SummaryWriter('./logs/summary')


def train(project_name, models_root, net, device, q, log_txt_file, val_datapath,
          models_train_best_coca, models_train_best_cosal, models_train_best_cosod, models_train_last,
          lr=1e-4, lr_de_epoch=25000, epochs=100000, log_interval=100, val_interval=1000):

    optimizer = Adam(net.parameters(), lr, weight_decay=1e-6)
    loss = Loss().cuda()
    ave_loss, ave_m_loss, ave_c_loss, ave_s_loss, ave_i_loss = 0, 0, 0, 0, 0
    coca_mae, cosal_mae, cosod_mae = 100, 100, 100
    for epoch in range(1, epochs + 1):
        img, cls_gt, mask_gt = q.get()
        net.zero_grad()
        img, cls_gt, mask_gt = img.cuda(), cls_gt.cuda(), mask_gt.cuda()

        pred_cls, pred_mask = net(img)
        all_loss, m_loss, c_loss, s_loss, iou_loss = loss(pred_mask, mask_gt, pred_cls, cls_gt)
        all_loss.backward()

        epoch_loss = all_loss.item()
        m_l = m_loss.item()
        c_l = c_loss.item()
        s_l = s_loss.item()
        i_l = iou_loss.item()
        ave_loss += epoch_loss
        ave_m_loss += m_l
        ave_c_loss += c_l
        ave_s_loss += s_l
        ave_i_loss += i_l
        optimizer.step()

        writer.add_scalar('All_Loss', all_loss.data, global_step=epoch)

        if epoch % log_interval == 0:
            ave_loss = ave_loss / log_interval
            ave_m_loss = ave_m_loss / log_interval
            ave_c_loss = ave_c_loss / log_interval
            ave_s_loss = ave_s_loss / log_interval
            ave_i_loss = ave_i_loss / log_interval
            custom_print(datetime.datetime.now().strftime('%F %T') +
                         ' lr: %e, epoch: [%d/%d], all_loss: [%.4f], m_loss: [%.4f], c_loss: [%.4f], s_loss: [%.4f], i_loss: [%.4f]' %
                         (lr, epoch, epochs, ave_loss, ave_m_loss, ave_c_loss, ave_s_loss, ave_i_loss), log_txt_file, 'a+')
            ave_loss, ave_m_loss, ave_c_loss, ave_s_loss, ave_i_loss = 0, 0, 0, 0, 0

        if epoch >= 40000 and epoch % val_interval == 0:
            net.eval()
            with torch.no_grad():
                custom_print(datetime.datetime.now().strftime('%F %T') + ' now is evaluating dataset', log_txt_file, 'a+')
                ave_mae = validation(net, val_datapath, device, group_size=5, img_size=224, img_dir_name='image',
                                     gt_dir_name='groundtruth', img_ext=['.jpg', '.jpg', '.jpg'],
                                     gt_ext=['.png', '.png', '.png'])

                writer.add_scalar('CoCA_MAE', torch.tensor(ave_mae[0]), global_step=epoch)
                writer.add_scalar('CoSal_MAE', torch.tensor(ave_mae[1]), global_step=epoch)
                writer.add_scalar('CoSOD_MAE', torch.tensor(ave_mae[2]), global_step=epoch)

                if ave_mae[0] < coca_mae:
                    torch.save(net.state_dict(), models_train_best_coca)
                    coca_mae = ave_mae[0]
                if ave_mae[1] < cosal_mae:
                    torch.save(net.state_dict(), models_train_best_cosal)
                    cosal_mae = ave_mae[1]
                if ave_mae[2] < cosod_mae:
                    torch.save(net.state_dict(), models_train_best_cosod)
                    cosod_mae = ave_mae[2]

                torch.save(net.state_dict(), models_train_last)

                custom_print('-' * 100, log_txt_file, 'a+')
                custom_print(datetime.datetime.now().strftime('%F %T') + ' CoCA  mae: [%.4f]' % (ave_mae[0]), log_txt_file, 'a+')
                custom_print(datetime.datetime.now().strftime('%F %T') + ' CoSal2015  mae: [%.4f]' % (ave_mae[1]), log_txt_file, 'a+')
                custom_print(datetime.datetime.now().strftime('%F %T') + ' CoSOD3k  mae: [%.4f]' % (ave_mae[2]), log_txt_file, 'a+')
                custom_print('-' * 100, log_txt_file, 'a+')

            net.train()

        if epoch % lr_de_epoch == 0:
            optimizer = Adam(net.parameters(), lr / 2, weight_decay=1e-6)
            lr = lr / 2
        writer.add_scalar('learning_rate', lr, global_step=epoch)

        if epoch >= 40000 and epoch % 2000 == 0:
            torch.save(net.state_dict(), models_root + '/' + project_name + '_epoch_{}.pth'.format(epoch))



def train_finetune(net, data_path, device, bs, log_txt_file, val_datapath, models_train_best_fbms, models_train_last,
                   lr=1e-4, lr_de_epoch=25000, epochs=100000, log_interval=100, val_interval=1000):
    optimizer = Adam(net.parameters(), lr, weight_decay=1e-6)
    train_loader = DataLoader(VideoDataset(data_path, epochs * bs, use_flow=False), num_workers=4,
                              batch_size=bs, shuffle=True, drop_last=False, pin_memory=False)
    loss = Video_Loss().cuda()
    ave_loss, ave_m_loss, ave_s_loss, ave_i_loss = 0, 0, 0, 0
    fbms_mae = 100
    epoch = 0
    for data, mask in train_loader:
        epoch += 1
        data = data.view(-1, data.shape[2], data.shape[3], data.shape[4])
        mask = mask.view(-1, mask.shape[2], mask.shape[3])
        img, mask_gt = data, mask
        net.zero_grad()
        img, mask_gt = img.cuda(), mask_gt.cuda()
        _, pred_mask = net(img)
        all_loss, m_loss, s_loss, iou_loss = loss(pred_mask, mask_gt)
        all_loss.backward()
        epoch_loss = all_loss.item()
        m_l = m_loss.item()
        s_l = s_loss.item()
        i_l = iou_loss.item()
        ave_loss += epoch_loss
        ave_m_loss += m_l
        ave_s_loss += s_l
        ave_i_loss += i_l
        optimizer.step()

        if epoch % log_interval == 0:
            ave_loss = ave_loss / log_interval
            ave_m_loss = ave_m_loss / log_interval
            ave_s_loss = ave_s_loss / log_interval
            ave_i_loss = ave_i_loss / log_interval
            custom_print(datetime.datetime.now().strftime('%F %T') +
                         ' lr: %e, epoch: [%d/%d], all_loss: [%.4f], m_loss: [%.4f], s_loss: [%.4f], i_loss: [%.4f]' %
                         (lr, epoch, epochs, ave_loss, ave_m_loss, ave_s_loss, ave_i_loss), log_txt_file, 'a+')
            ave_loss, ave_m_loss, ave_s_loss, ave_i_loss = 0, 0, 0, 0

        if epoch % val_interval == 0:
            net.eval()
            with torch.no_grad():
                custom_print(datetime.datetime.now().strftime('%F %T') +
                             ' now is evaluating the coseg dataset', log_txt_file, 'a+')
                ave_mae = validation(net, val_datapath, device, group_size=10, img_size=224, img_dir_name='image',
                                     gt_dir_name='groundtruth', img_ext=['.jpg', '.jpg', '.jpg'],
                                     gt_ext=['.png', '.png', '.png'])

                writer.add_scalar('FBMS_MAE', torch.tensor(ave_mae[0]), global_step=epoch)

                if ave_mae[0] < fbms_mae:
                    torch.save(net.state_dict(), models_train_best_fbms)
                    fbms_mae = ave_mae[0]

                torch.save(net.state_dict(), models_train_last)

                custom_print('-' * 100, log_txt_file, 'a+')
                custom_print(datetime.datetime.now().strftime('%F %T') + ' FBMS  mae: [%.4f]' % (ave_mae[0]), log_txt_file, 'a+')
                custom_print('-' * 100, log_txt_file, 'a+')

            net.train()

        if epoch % lr_de_epoch == 0:
            optimizer = Adam(net.parameters(), lr / 2, weight_decay=1e-6)
            lr = lr / 2

