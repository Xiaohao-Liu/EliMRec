import os
import sys
import time
import torch
import collections
import numpy as np
from torch import optim


# import dataloader
from data.dataset import Dataset
from models import *
from util import set_seed
from util.logger import Logger
from util.meter import Meter
from util.configurator import Configurator
from data import PairwiseSamplerV2, PointwiseSamplerV2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

class Net:
    def __init__(self, args):
        self.config = args
        Logger.logger = Logger(
            name="_".join([self.config.recommender, self.config["data.input.dataset"],self.config.loss, self.config.suffix]),
            show_in_console=False if self.config.verbose == 0 else True,
            is_creat_log_file=self.config.create_log_file,
            path=self.config.log_path)
        Logger.info(self.config.params_str())
        print("use","cuda:0" if torch.cuda.is_available() else "cpu")
        self.config.device = torch.device("cuda:0" if torch.cuda.is_available() and not self.config.no_cuda else "cpu")
        self.sw_lock = False
        self.dataset = Dataset(self.config)
        
        self.cf_mode = True
        
        # define model
        if self.config.recommender == 'EliMRec':
            self.recommender = EliMRec(self.config, self.dataset).to(self.config.device)
            self.cf_mode = True if 'cf_mode' not in self.config else self.config['cf_mode']

        Logger.info(get_parameter_number(self.recommender))

        self.opt = optim.Adam(self.recommender.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

    def run(self):
        topk = str(self.config['topks'][0])
        meter_id = int(time.time())
        loss_meter = Meter("loss", id=meter_id)
        recall_meter = Meter("recall@" + topk, id=meter_id)
        precision_meter = Meter("precision@" + topk,  id=meter_id)
        ndcg_meter = Meter("ndcg@" + topk, id=meter_id)
        
        data_iter = PairwiseSamplerV2(self.dataset, neg_num=1, batch_size=self.config.batch_size, shuffle=True)
        hasNeg = True
        best_recall = 0
        best_epoch = 0
        best_result = ""
        final_result = ""
        if self.cf_mode:
            best_recall = {'TE':0, 'TIE':0}
            best_epoch = {'TE':0, 'TIE':0}
            best_result = {'TE':"", 'TIE':""}
            final_result = {'TE':"", 'TIE':""}
            recall_meter = {}
            precision_meter = {}
            ndcg_meter = {}
            for effect in ['TIE']:
                recall_meter[effect] = Meter("R@" + topk, id=meter_id)
                precision_meter[effect] = Meter("P@" + topk,  id=meter_id)
                ndcg_meter[effect] =  Meter("N@" + topk, id=meter_id)

        model_name = self.recommender.getFileName()
        
        for epoch in range(self.config.num_epoch):
            self.recommender.train()
            Logger.info('======================')
            Logger.info(f'EPOCH[{epoch}/{self.config.num_epoch}]')
            start = time.time()
            loss_meter.reset_time()

            batch_size = self.config.batch_size if self.config.batch_size != -1 else len(users)

            #  Meters
            batch_loss_meter = Meter(name="MultiLoss(bpr)")
            batch_loss_meter.reset()
            for bat_users, bat_pos_items, bat_neg_items in tqdm(data_iter):

                bat_users = torch.tensor(bat_users).to(self.config.device)
                bat_pos_items = torch.tensor(bat_pos_items).to(self.config.device)
                bat_neg_items = torch.tensor(bat_neg_items).to(self.config.device)
                batch_loss_meter.reset_time()
                loss = self.recommender.__getattribute__(self.config.loss)(bat_users, bat_pos_items, bat_neg_items)
                self.opt.zero_grad()
                loss.backward(retain_graph=True)
                self.opt.step()
                batch_loss_meter.update(val=loss.cpu().item())

            if self.config.recommender in ['MCN']:
                loss = self.recommender.epoch_loss()
                self.opt.zero_grad()
                loss.backward(retain_graph=True)
                self.opt.step()
                Logger.info("Epoch Loss: {}".format(loss.cpu().item()))
            # test
            if (epoch + 1) % self.config["test_step"] == 0:
                Logger.info('[VALID]')
                self.recommender.eval()
                Logger.info('[CF Mode]')
                # evaluate
                effect = 'TIE'
                recall_meter[effect].reset_time()
                precision_meter[effect].reset_time()
                ndcg_meter[effect].reset_time()
                # set type for recommender
                self.recommender.predict_type = effect
                current_result, buf = self.recommender.evaluate()
                if current_result is not None:
                    recall_meter[effect].update(val=current_result[1], epoch=epoch)
                    precision_meter[effect].update(val=current_result[0], epoch=epoch)
                    ndcg_meter[effect].update(val=current_result[2], epoch=epoch)
                    Logger.info("[{}]\t{}\t{}\t{}".format(effect,recall_meter[effect], ndcg_meter[effect], precision_meter[effect]))

                # test
                if recall_meter["TIE"].val > best_recall["TIE"] and epoch != 0:
                    if self.config["save_flag"]:
                        Logger.info(f'[saved][EPOCH {epoch}]')
                        torch.save(self.recommender.state_dict(), model_name)
                    best_result["TIE"] = "[EPOCH {}]\n{}\t{}\t{}".format(epoch, recall_meter[effect], ndcg_meter[effect], precision_meter[effect])
                    Logger.info("[Better Result]")
                    Logger.info('[TEST]')
                    for effect in ['TE','TIE']:
                        best_recall[effect] = current_result[1]
                        best_epoch[effect] = epoch
                        self.recommender.predict_type = effect
                        test_result, _ = self.recommender.test()
                        assert test_result is not None
                        final_result[effect] = "  [{}]\t{}\t{}\t{}".format(effect,test_result[1],test_result[0],test_result[2])
                        Logger.info(final_result[effect])
                if (epoch - best_epoch["TIE"]) > self.config.stop_cnt:
                    # stop training
                    break

            loss_ = batch_loss_meter.avg
            loss_meter.update(val=loss_, epoch=epoch)
            Logger.info(f'{loss_meter}')
#         finally:
        if best_recall != 0:
            for effect in ['TIE']:
                Logger.info("=>[{}] best_valid_result:\n{}".format(effect, best_result[effect]))
            for effect in ['TE','TIE']:
                Logger.info("=>[{}] test_result:\n{}".format(effect, final_result[effect]))
        

if __name__ == '__main__':

    # load cofig params
    is_windows = sys.platform.startswith('win')
    if is_windows:
        root_folder = 'E:/pycharmprojects/projects/backupcode/'
    else:
        root_folder = '/home/lxh/Liuxiaohao/new_rec_bias/'#'/workspace/Liuxiaohao/rec_bias/'

    args = Configurator(root_folder + "NeuRec.properties", default_section="hyperparameters")
    
    # set radom seed
    set_seed(args["seed"])
    # start
    egcn = Net(args)
    egcn.run()
