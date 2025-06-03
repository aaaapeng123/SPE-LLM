from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, adjust_model
from utils.metrics import metric
import torch
import torch.nn as nn
from models import S2IPLLM
from torch.nn.utils import clip_grad_norm_
from utils.losses import mape_loss, mase_loss, smape_loss

from transformers import AdamW
# from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch import optim
import os
import time
import warnings
import numpy as np

from tqdm import tqdm

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'S2IPLLM': S2IPLLM,
        }
        self.device = torch.device('cuda')
        self.model = self._build_model()
        # 初始化 Accelerator
        self.train_data, self.train_loader = self._get_data(flag='train')
        self.vali_data, self.vali_loader = self._get_data(flag='val')
        # self.test_data, self.test_loader = self._get_data(flag='test')

        self.optimizer = self._select_optimizer()
        self.criterion = self._select_criterion()

    def print_and_save_model_params(self, model, file_path):
        """
        打印模型的所有参数信息，并保存到指定文件
        """
        # 打开文件进行写入
        with open(file_path, 'w') as f:
            for name, param in model.named_parameters():
                # 计算参数的大小（总元素数量），并转换为百万为单位
                param_size_in_m = param.numel() / 1e6  # 转换为百万
                # 打印并写入文件
                f.write(f"Parameter name: {name}\n")
                f.write(f"Size (M): {param_size_in_m:.6f}M\n")  # 格式化为百万
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).to(self.device)
        model_params_file = '/home/u202332803043/ljp/S2IP-LLM-main/Long-term_Forecasting/para/wea_model_params.txt'
        # self.print_and_save_model_params(model, model_params_file)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'MSE':
            criterion = nn.MSELoss()
        elif self.args.loss == 'SMAPE':
            criterion = smape_loss()
        elif self.args.loss == 'MAE':
            criterion = nn.L1Loss()  # PyTorch 的 L1Loss 即为 MAE
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(
                    torch.bfloat16).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs,res = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs, res = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:self.args.number_variable].float().to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:self.args.number_variable].float().to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            simlarity_losses = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(self.train_loader)):
                iter_count += 1
                self.optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # 修改损失函数地方定义频域损失权重
                alpha = 0.1  # 可根据需要调整, 0 <= alpha <= 1
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs,res = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:self.args.number_variable]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:self.args.number_variable].float().to(self.device)
                        #####原loss计算
                        # loss = self.criterion(outputs, batch_y)
                        #####原loss计算
                        ####更改后混合loss计算
                        # 原始时域损失
                        loss_tmp = self.criterion(outputs, batch_y)
                        # 计算频域损失
                        loss_feq = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs().mean()
                        # 总损失为时域损失与频域损失的加权组合
                        loss = alpha * loss_tmp + (1 - alpha) * loss_feq
                        #####更改后混合loss计算
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs, res = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:self.args.number_variable]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:self.args.number_variable].float().to(self.device)
                    #####原loss计算
                    # loss = self.criterion(outputs, batch_y)
                    #####原loss计算
                    ####更改后混合loss计算
                    # 原始时域损失
                    loss_tmp = self.criterion(outputs, batch_y)
                    # 计算频域损失
                    loss_feq = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs().mean()
                    # 总损失为时域损失与频域损失的加权组合
                    loss = alpha * loss_tmp + (1 - alpha) * loss_feq
                    #####更改后混合loss计算
                    train_loss.append(loss.item())

                    #更改前loss
                    # simlarity_losses.append(res['simlarity_loss'].item())
                    #更改后loss
                    simlarity_losses.append(res['simlarity_loss'])



                    loss += self.args.sim_coef * res['simlarity_loss']

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    # Loss Scale
                    scaler.scale(loss).backward()
                    # 更新参数
                    scaler.step(self.optimizer)
                    # 更新scaler
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            sim_loss = np.average(simlarity_losses)
            vali_loss = self.vali(self.vali_data, self.vali_loader, self.criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Sim Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, sim_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.optimizer, epoch + 1, self.args)
            adjust_model(self.model, epoch + 1, self.args)

    # def test(self, setting, test=1):
    #
    #     test_data, test_loader = self._get_data(flag='test')
    #
    #     preds = []
    #     trues = []
    #     folder_path = './test_results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #
    #     sim_matrix = []
    #     input_embedding = []
    #     prompted_embedding = []
    #     last_embedding = []
    #
    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float().to(self.device)
    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)
    #             # decoder input
    #             dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
    #             dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
    #             # encoder - decoder
    #             if self.args.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     if self.args.output_attention:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                     else:
    #                         outputs,res = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #             else:
    #                 if self.args.output_attention:
    #                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                 else:
    #                     outputs, res = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #             f_dim = -1 if self.args.features == 'MS' else 0
    #             outputs = outputs[:, -self.args.pred_len:,
    #                       f_dim:self.args.number_variable].float().detach().cpu().numpy()
    #             batch_y = batch_y[:, -self.args.pred_len:,
    #                       f_dim:self.args.number_variable].float().detach().cpu().numpy()
    #             pred = outputs
    #             true = batch_y
    #             preds.append(pred)
    #             trues.append(true)
    #             if i % 20 == 0:
    #                 input = batch_x.float().detach().cpu().numpy()
    #                 gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
    #                 pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
    #                 visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
    #     preds = np.array(preds)
    #     trues = np.array(trues)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    #     trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    #     print('test shape:', preds.shape, trues.shape)
    #
    #     # result save
    #     folder_path = './results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #
    #     mae, mse, rmse, mape, mspe = metric(preds, trues)
    #     print('mse:{}, mae:{}'.format(mse, mae))
    #     f = open("result_long_term_forecast.txt", 'a')
    #     f.write(setting + "  \n")
    #     f.write('mse:{}, mae:{}'.format(mse, mae))
    #     f.write('\n')
    #     f.write('\n')
    #     f.close()
    #
    #     np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    #     np.save(folder_path + 'pred.npy', preds)
    #     np.save(folder_path + 'true.npy', trues)
    #     return mse, mae
    ###可视化最好的图
    def test(self, setting, test=1):
        test_data, test_loader = self._get_data(flag='test')
        preds = []
        trues = []
        inputs = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # model forward
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:self.args.number_variable]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:self.args.number_variable]

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                inp = batch_x.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                inputs.append(inp)

                # 每隔20步可视化一个样本
                if i % 20 == 0:
                    gt = np.concatenate((inp[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((inp[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # 拼接所有数据
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputs = np.concatenate(inputs, axis=0)

        print('test shape:', preds.shape, trues.shape)

        # 保存结果
        result_folder = './results/' + setting + '/'
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n\n')

        np.save(result_folder + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(result_folder + 'pred.npy', preds)
        np.save(result_folder + 'true.npy', trues)

        # === 新增：找出误差最小的样本进行可视化 ===
        errors = np.mean((preds[:, :, -1] - trues[:, :, -1]) ** 2, axis=1)  # shape: (B,)
        best_idx = np.argmin(errors)

        input_target = inputs[best_idx, :, -1]
        true_target = trues[best_idx, :, -1]
        pred_target = preds[best_idx, :, -1]

        gt = np.concatenate((input_target, true_target), axis=0)
        pd = np.concatenate((input_target, pred_target), axis=0)

        # Visualize the best sample
        visual(gt, pd, os.path.join(folder_path, 'best_sample.pdf'))

        # Save the best sample's true and predicted values
        np.save(os.path.join(folder_path, 'best_sample_true.npy'), true_target)
        np.save(os.path.join(folder_path, 'best_sample_pred.npy'), pred_target)
        np.save(os.path.join(folder_path, 'best_sample_input.npy'), input_target)

        # Optionally, save the error for the best sample
        best_error = errors[best_idx]
        np.save(os.path.join(folder_path, 'best_sample_error.npy'), best_error)

        return mse, mae

#