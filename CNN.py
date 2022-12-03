import base64
import os
import json
import re

from common.logUtil import logUtil
import sys
from model.service import cnnService as model_service
from torchvision import datasets


class CNN:

    def __init__(self, argv):
        super(CNN, self).__init__()

        base64_str = argv[1]
        base64_str = base64_str + '=' * (4 - len(base64_str) % 4) if len(base64_str) % 4 != 0 else base64_str
        base64_bytes = base64.b64decode(base64_str)
        param_str = base64_bytes.decode('ascii')

        param_json = json.loads(param_str)

        # parameter setting
        workflow_no = param_json['workflow_no']  # '1647936458825'
        workflow_item_no = param_json['workflow_item_no']  # 'm_DecisionTreeClassifier_100006'
        model_file = param_json['model_file']  # 'D:/workspace/e-Mapp/model/m_DecisionTreeClassifier_100006.pth'
        csv_file = param_json['csv_file']
        out_file = param_json['out_file']
        tree_file = param_json['tree_file']
        model_attr = param_json['model_attr']
        encoding = param_json['encoding']
        debug = param_json['debug']

        self.log = logUtil(debug)
        self.encoding = encoding
        self.workflow_no = workflow_no
        self.workflow_item_no = workflow_item_no
        self.model_file = model_file
        self.model_attr = model_attr.replace('\'', '\"')
        self.base_dir = 'D:/workspace/stat.analy/src/eMapp-data/indata'  # 분류 항목
        self.csv_file = csv_file  # 'D:/workspace/e-Mapp/tmp/1647933908298_m_SimpleSel_100004_20220324150522.csv'
        self.outfile = out_file  # 'D:/workspace/e-Mapp/tmp/1647933908298_m_SimpleSel_100004_20220324150522_predict.csv'
        self.tree_file = tree_file  # 'D:/workspace/e-Mapp/tmp/1647933908298_m_SimpleSel_100004_20220324150522_tree.csv'

        self.log.print('{0}-{1} CNN 속성 설정 완료'.format(self.workflow_no, self.workflow_item_no))
        self.log.print('{0}-{1}\n{2}'.format(self.workflow_no, self.workflow_item_no, base64_str))

    def execute(self):

        self.log.print('{0}-{1} decisionTreeClassifier 시작'.format(self.workflow_no, self.workflow_item_no))

        # Json 타입으로 변환한다.
        algorithm_json = json.loads(self.model_attr)

        # json process
        key_list = []
        value_list = []

        reg_exp = "[-+]?\d+[. ]?\d?"
        for i in range(0, len(algorithm_json['rows'])):
            key_list.append(algorithm_json['rows'][i]["name"])
            # 파라미터 값이 None이거나 숫자인 경우
            if (algorithm_json['rows'][i]["value"]) == 'None' or re.match(reg_exp,
                                                                          algorithm_json['rows'][i]["value"]):
                value_list.append(eval(algorithm_json['rows'][i]["value"]))
                # 파라미터 값이 String인 경우
            else:
                value_list.append(algorithm_json['rows'][i]["value"])
        params = dict(zip(key_list, value_list))

        # 기본 경로
        train_dir = params['train_dir']
        test_dir = params['test_dir']
        predict_dir = params['predict_dir']
        classes = params['classes']

        exec_mode = params['exec_mode']
        image_height = params['image_height']
        image_width = params['image_width']
        meanRGB = params['meanRGB']
        stdRGB = params['stdRGB']
        opt = params['opt']

        service = model_service.cnnService(exec_mode, image_height, image_width, train_dir, test_dir, predict_dir,
                                           meanRGB, stdRGB, opt, classes)

        # converting the list to numpy array

        if exec_mode == 'auto':
            # 생성된 모델이 있는지 확인
            if os.path.isfile(self.model_file):
                os.remove(self.model_file)
            exec_mode_list = ['train', 'eval']
        else:
            exec_mode_list = exec_mode.split()

        for exec_mode in exec_mode_list:
            # train
            if exec_mode == 'train':
                # 기존 모델 가져 오기
                model = service.load_model(self.model_file)
                if not isinstance(None, type(model)):
                    service.set_model(model)
                else:
                    # 모델 생성
                    service.create_model()

                # 데이터셋 전달 (train,validate,test 포함)
                service.set_data(datasets)

                # training
                self.log.print('training')

                my_model = service.training(5)

                # model save
                service.save_model(my_model, self.model_file)

                # 모델 닫기
                service.close_model()

            elif exec_mode == 'test':
                # 기존 모델 가져오기
                model = service.load_model(self.model_file)
                if not isinstance(None, type(model)):
                    service.set_model(model)

                    # 학습된 모델에 대해서 테스트 진행
                    service.test()
                    # 모델 닫기
                    service.close_model()

                else:
                    # 모델이 존재하지 않음
                    self.log.print('{0} 모델 화일이 없습니다. train부터 진행해 주세요.'.format(self.model_file))

            elif exec_mode == 'eval':
                # 기존 모델 가져오기
                model = service.load_model(self.model_file)
                service.set_model(model)

                # 모델 닫기
                service.close_model()

            self.log.print('{0}-{1} Cnn {2} 완료'
                           .format(self.workflow_no, self.workflow_item_no, exec_mode))


def main():
    # 모델을 실행 시키기 위한 클래스를 생성한다.
    workflow_model = CNN(sys.argv)
    # 모델을 실행시킨다.
    workflow_model.execute()


if __name__ == "__main__":
    main()
