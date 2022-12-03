# -*- encoding: utf-8 -*-
import base64
import json
import os
import re
import sys
from datetime import datetime
import pandas as pd

from common.logUtil import logUtil
from common.datasetModule import DatasetSplit_ML as DatasetSplit_ML
from model.service import linearRegressionService as model_service


class linearRegressor:


    def __init__(self, argv):

        super(linearRegressor, self).__init__()

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
        self.csv_file = csv_file  # 'D:/workspace/e-Mapp/tmp/1647933908298_m_SimpleSel_100004_20220324150522.csv'
        self.outfile = out_file  # 'D:/workspace/e-Mapp/tmp/1647933908298_m_SimpleSel_100004_20220324150522_predict.csv'
        self.tree_file = tree_file  # 'D:/workspace/e-Mapp/tmp/1647933908298_m_SimpleSel_100004_20220324150522_tree.csv'

        self.log.print('{0}-{1} linearRegressor 속성 설정 완료'.format(self.workflow_no, self.workflow_item_no))
        self.log.print('{0}-{1}\n{2}'.format(self.workflow_no, self.workflow_item_no, base64_str))

    def execute(self):

        self.log.print('{0}-{1} linearRegressor 시작'.format(self.workflow_no, self.workflow_item_no))

        # 재귀를 사용
        sys.setrecursionlimit(10 ** 7)

        # Json 타입으로 변환한다.
        algorithm_json = json.loads(self.model_attr)

        # json process
        key_list = []
        value_list = []
        reg_exp = "[-+]?\d+[. ]?\d?"
        for i in range(0, len(algorithm_json['rows'])):
            key_list.append(algorithm_json['rows'][i]["name"])
            # 파라미터 값이 None이거나 숫자인 경우
            if (algorithm_json['rows'][i]["value"]) == 'None' or re.match(reg_exp, algorithm_json['rows'][i]["value"]):
                value_list.append(eval(algorithm_json['rows'][i]["value"]))
            # 파라미터 값이 String인 경우
            else:
                value_list.append(algorithm_json['rows'][i]["value"])
        params = dict(zip(key_list, value_list))

        # 분석할 데이터를 읽어 온다.
        df = pd.read_csv(self.csv_file, encoding=self.encoding)
        self.log.print(df)
        df
        # parameter setting
        exec_mode = params['exec_mode']
        evaluation_mode = params['evaluation_mode']
        n_splits = params['n_splits']

        # Params Setting
        fit_intercept = params['fit_intercept']
        if fit_intercept == str(True):
            fit_intercept = bool(1)
        elif fit_intercept == str(False):
            fit_intercept = bool(0)
        normalize = params['normalize']
        if normalize == str(False):
            normalize = bool(0)
        elif normalize == str(True):
            normalize = bool(1)
        copy_X = params['copy_X']
        n_jobs = params['n_jobs']
        positive = params['positive']
        # 데이터 설정
        # x_date, y_data를 설정하기 위한 y값 설정
        y_data_item = params['y_data']
        # 테스트데이터 샘플률
        test_split = float(params['test_split'])
        # 검증데이터 샘플률
        val_split = float(params['val_split'])
        # 최종 결과값 저장 키값
        ids_items = params['ids']
        ids = ids_items.split(',')
        # x_data 설정
        x_data = df.drop([y_data_item], axis=1)
        # y_data 설정
        y_data = df[y_data_item]

        # regression Service 생성
        service = model_service.linearRegressionService(evaluation_mode, fit_intercept, normalize, copy_X, n_jobs,
                                                        positive, n_splits, self.log)

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

                # 데이터셋분리 (train, validation, test).
                train_data_split = DatasetSplit_ML(x_data, y_data, test_split, val_split)

                # 데이터셋분리
                datasets = train_data_split.get_train_val_test_dataset()

                # 데이터셋 전달 (train,validate,test 포함)
                service.set_data(datasets)

                # training
                self.log.print('training')

                my_model = service.training()

                # model save
                service.save_model(my_model, self.model_file)
                # 모델 닫기
                service.close_model()

            elif exec_mode == 'test':
                # 기존 모델 가져오기
                model = service.load_model(self.model_file)
                if not isinstance(None, type(model)):
                    service.set_model(model)
                    # 데이터셋분리 (train, validation, test).
                    train_data_split = DatasetSplit_ML(x_data, y_data, test_split, val_split)

                    # 데이터셋분리
                    datasets = train_data_split.get_train_val_test_dataset()

                    # 데이터셋 전달 (train,validate,test 포함)
                    service.set_data(datasets)

                    # 학습된 모델에 대해서 테스트 진행
                    service.test()
                    # 모델 닫기
                    service.close_model()
                else:
                    # 모델 생성
                    service.create_model()

            elif exec_mode == 'eval':
                # 기존 모델 가져오기
                model = service.load_model(self.model_file)
                service.set_model(model)

                # 결과를 저장하기 위한 데이터프레임
                prediction = service.prediction(x_data.values.tolist())
                create_time = datetime.today().strftime("%Y%m%d%H%M%S")

                # 결과값을 저장하기 위한 변수 선연
                result_data = {}
                for iid in ids:
                    result_data[iid] = x_data[iid].values.tolist()
                result_data['prediction'] = prediction
                result_data['createtime'] = create_time
                result_df = pd.DataFrame(result_data)
                result_df['prediction'] = result_df['prediction'].replace([False], 0).replace([True], 1)
                self.log.print(result_df)

                result_df.to_csv(self.outfile, sep=',', na_rep='NaN', index=False)

                # 모델 닫기
                service.close_model()

        self.log.print('{0}-{1} linearRegressor {2} 완료'.format(self.workflow_no, self.workflow_item_no, exec_mode))


def main():
    # 모델을 실행 시키기 위한 클래스를 생성한다.
    workflow_model = linearRegressor(sys.argv)
    # 모델을 실행시킨다.
    workflow_model.execute()


if __name__ == "__main__":
    main()
