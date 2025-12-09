# RTIOD

# Analysis Report

## Root

| File | Path | Description |
| :--- | :--- | :--- |
| Bicycle&Motorcycle.py | [RTIOD\Bicycle&Motorcycle.py](file:///C:/junha/Git/RTIOD/Bicycle&Motorcycle.py) | 이 파일은 JSON 파일에서 자전거와 모터사이클의 이미지 데이터를 추출하여 각 월에 따른 이미지 개수를 그래프로 시각화하는 코드를 구현합니다. |

## Prediction_Validation

| File | Path | Description |
| :--- | :--- | :--- |
| challenge_evaluator.py | [RTIOD\Prediction_Validation\challenge_evaluator.py](file:///C:/junha/Git/RTIOD/Prediction_Validation/challenge_evaluator.py) | 이 코드는 라보스테이블 열 이미지 객체 탐지 도전Evaluator 스크립트로, 예측 파일과 참조 파일을 로드하여 평가 점수를 계산하고 출력하는 기능을 포함합니다. |

## WorkStation_AuxDet

| File | Path | Description |
| :--- | :--- | :--- |
| AuxDetScratch.py | [RTIOD\WorkStation_AuxDet\Model\AuxDetScratch.py](file:///C:/junha/Git/RTIOD/WorkStation_AuxDet/Model/AuxDetScratch.py) | 이 파일은 객체 인식 모델의 辅助检测 (보조 감지) 기능을 구현하기 위한 파이토치 클래스입니다. |
| M2DM.py | [RTIOD\WorkStation_AuxDet\Model\M2DM.py](file:///C:/junha/Git/RTIOD/WorkStation_AuxDet/Model/M2DM.py) | 이 파일은 딥러닝 모델에서 동적 프로젝션과 공간적 모듈러스 기능을 구현하기 위한 PyTorch 클래스입니다. |
| RPN_ROI_Heads.py | [RTIOD\WorkStation_AuxDet\Model\RPN_ROI_Heads.py](file:///C:/junha/Git/RTIOD/WorkStation_AuxDet/Model/RPN_ROI_Heads.py) | 이 파일은 Region-based Convolutional Neural Network의 RPN(Region Proposal Network)과 ROI Heads를 구현한 코드로, 객체 탐지 모델의 일부분을 포함합니다. |
| Inference.py | [RTIOD\WorkStation_AuxDet\Inference.py](file:///C:/junha/Git/RTIOD/WorkStation_AuxDet/Inference.py) | 이 파일은 객체 탐지 모델을 사용하여 이미지에서 객체를 인식하고 검출 결과를 시각화하는 코드를 구현합니다. |
| IRDataset.py | [RTIOD\WorkStation_AuxDet\IRDataset.py](file:///C:/junha/Git/RTIOD/WorkStation_AuxDet/IRDataset.py) | 이 파일은 이미지 데이터셋을 구현한 클래스를 정의하며, CSV 파일과 이미지 폴더에서 데이터를 로드하고 처리합니다. |
| main.py | [RTIOD\WorkStation_AuxDet\main.py](file:///C:/junha/Git/RTIOD/WorkStation_AuxDet/main.py) | 이 파일은 객체 탐지 모델을 학습하고 평가하는 스크립트입니다. |
| Test_Step.py | [RTIOD\WorkStation_AuxDet\Test_Step.py](file:///C:/junha/Git/RTIOD/WorkStation_AuxDet/Test_Step.py) | 코드는 텐서플로우(torch)를 사용하여 모델의 추론 단계(test_step)를 구현한 함수입니다. |
| Train_Step.py | [RTIOD\WorkStation_AuxDet\Train_Step.py](file:///C:/junha/Git/RTIOD/WorkStation_AuxDet/Train_Step.py) | 이 파일은 훈련 단계 함수를 정의하며, 데이터로더로부터 입력받은 이미지와 메타데이터를 모델에 전달하여 손실을 계산하고 최적화 단계를 수행합니다. |
| Utils.py | [RTIOD\WorkStation_AuxDet\Utils.py](file:///C:/junha/Git/RTIOD/WorkStation_AuxDet/Utils.py) | 이 파일은 객체 검출 모델 평가를 위한 도구함수들을 정의합니다. 주요 기능으로는 IOU 계산, 미니멈 평균 정확도 계산, 데이터로더와 모델을 이용한 평가 수행 등이 포함되어 있습니다. |

## WorkStation_FasterRCNN

| File | Path | Description |
| :--- | :--- | :--- |
| IRNoMetaDataset.py | [RTIOD\WorkStation_FasterRCNN\IRNoMetaDataset.py](file:///C:/junha/Git/RTIOD/WorkStation_FasterRCNN/IRNoMetaDataset.py) | 이 파일은 인코딩된 이미지 데이터셋을 처리하는 PyTorch Dataset 클래스를 정의합니다. |
| main_FasterRCNN.py | [RTIOD\WorkStation_FasterRCNN\main_FasterRCNN.py](file:///C:/junha/Git/RTIOD/WorkStation_FasterRCNN/main_FasterRCNN.py) | 이 파일은 Faster RCNN 모델을 사용하여 COCO 데이터셋에서 객체 검출을 수행하는 PyTorch 구현을 포함합니다. |
| Train_Step.py | [RTIOD\WorkStation_FasterRCNN\Train_Step.py](file:///C:/junha/Git/RTIOD/WorkStation_FasterRCNN/Train_Step.py) | 이 파일은 텐서플로우를 사용하여 모델의 훈련 단계를 정의합니다. |
| Utils.py | [RTIOD\WorkStation_FasterRCNN\Utils.py](file:///C:/junha/Git/RTIOD/WorkStation_FasterRCNN/Utils.py) | 이 파일은 객체 탐지 모델을 평가하기 위한 함수들을 정의합니다. |

## WorkStation_MoE

| File | Path | Description |
| :--- | :--- | :--- |
| MMMMoE.py | [RTIOD\WorkStation_MoE\MMMMoE\Original_MMMMoE\MMMMoE.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE/MMMMoE/Original_MMMMoE/MMMMoE.py) | 이 파일은 머신러닝 모델의 객체 탐지 기능을 구현합니다. |
| MoEBlock.py | [RTIOD\WorkStation_MoE\MMMMoE\Original_MMMMoE\MoEBlock.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE/MMMMoE/Original_MMMMoE/MoEBlock.py) | 이 파일은 메타데이터를 기반으로 동적 전문가 선택을 수행하는 MoEBlock 클래스와 그에 따른 가이트 모듈을 정의합니다. |
| RPN_ROI_Head.py | [RTIOD\WorkStation_MoE\MMMMoE\Original_MMMMoE\RPN_ROI_Head.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE/MMMMoE/Original_MMMMoE/RPN_ROI_Head.py) | 이 파일은 Faster R-CNN 모델의 RPN과 ROI 헤드를 구축하는 함수를 정의합니다. |
| Backbone.py | [RTIOD\WorkStation_MoE\MMMMoE\Backbone.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE/MMMMoE/Backbone.py) | 이 파일은 PyTorch를 사용하여 다양한 ResNet 아키텍처를 기반으로 이미지 추출기(Image Extractor) 클래스를 정의합니다. |
| MMMMoE.py | [RTIOD\WorkStation_MoE\MMMMoE\MMMMoE.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE/MMMMoE/MMMMoE.py) | 이 코드는 객체 검출 모델로 MMMPoE_Detector 클래스를 정의하며, 이 클래스는 Feature Pyramid Network와 MoEBlock을 포함한 복잡한 구조를 가지고 있습니다. |
| MoEBlock.py | [RTIOD\WorkStation_MoE\MMMMoE\MoEBlock.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE/MMMMoE/MoEBlock.py) | 이 파일은 MoEBlock 클래스를 정의하며, 이를 통해 복잡한 머신 러닝 모델의 효율성을 높이는 기능을 구현합니다. |
| RPN_ROI_Head.py | [RTIOD\WorkStation_MoE\MMMMoE\RPN_ROI_Head.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE/MMMMoE/RPN_ROI_Head.py) | 이 코드는 Faster R-CNN 모델의 RPN(Region Proposal Network)과 RoI Head를 구축하는 함수를 정의합니다. |
| Data_Check.py | [RTIOD\WorkStation_MoE\Preprocess\Data_Check.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE/Preprocess/Data_Check.py) | 이 파일은 데이터셋에서 사용되지 않은 라벨 파일을 확인하는 기능을 구현합니다. |
| Dataloader_Check.py | [RTIOD\WorkStation_MoE\Preprocess\Dataloader_Check.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE/Preprocess/Dataloader_Check.py) | 이 파일은 DataLoader를 검증하기 위해 LTDv2 데이터셋을 로드하고, 데이터 샘플의 존재 여부와 이미지 정보 등을 출력하는 코드입니다. |
| Json_Splitter.py | [RTIOD\WorkStation_MoE\Preprocess\Json_Splitter.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE/Preprocess/Json_Splitter.py) | 이 파일은 JSON 데이터를 로드하고 이미지와 애노테이션을 지정된 비율로 분할하여mini Train.json과 mini Test.json 파일로 저장합니다. |
| Train_Val_Splitter.py | [RTIOD\WorkStation_MoE\Preprocess\Train_Val_Splitter.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE/Preprocess/Train_Val_Splitter.py) | 이 파일은 JSON 형식의 이미지 데이터셋을 8:2의 비율로 훈련 세트와 검증 세트로 분할하는 코드입니다. |
| Inference.py | [RTIOD\WorkStation_MoE\Inference.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE/Inference.py) | Inference.py은 .py 파일입니다. 함수를 포함합니다: run_inference_and_visualize. |
| IRJsonDataset.py | [RTIOD\WorkStation_MoE\IRJsonDataset.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE/IRJsonDataset.py) | 이 파일은 JSON 형식의 데이터셋을 처리하는 PyTorch Dataset 클래스를 정의합니다. |
| main.py | [RTIOD\WorkStation_MoE\main.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE/main.py) | 이 파일은 객체 탐지 모델을 훈련시키는 코드이며, WarmupScheduler와 데이터로더 설정 등을 포함합니다. |
| Test_Step.py | [RTIOD\WorkStation_MoE\Test_Step.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE/Test_Step.py) | 이 파일은 텐서플로우를 사용하여 모델의 단계별 테스트를 구현합니다. |
| Train_Step.py | [RTIOD\WorkStation_MoE\Train_Step.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE/Train_Step.py) | 이 파일은 딥러닝 모델의 단계별 학습을 구현합니다. |
| Utils.py | [RTIOD\WorkStation_MoE\Utils.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE/Utils.py) | 이 파일은 객체 검출 모델을 평가하기 위한 함수들을 정의합니다. |
| Validation.py | [RTIOD\WorkStation_MoE\Validation.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE/Validation.py) | 이 파일은 머신러닝 모델을 검증하는 함수를 정의하고 해당 함수를 실행합니다. |

## WorkStation_MoE_3Channel

| File | Path | Description |
| :--- | :--- | :--- |
| Backbone.py | [RTIOD\WorkStation_MoE_3Channel\MMMMoE\Backbone.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE_3Channel/MMMMoE/Backbone.py) | 이 파일은 PyTorch를 사용해 Resnet18, Resnet34, Resnet50 모델의 초성 추출기를 구현한 코드입니다. |
| MMMMoE.py | [RTIOD\WorkStation_MoE_3Channel\MMMMoE\MMMMoE.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE_3Channel/MMMMoE/MMMMoE.py) | 이 파일은 머신러닝 모델의 디텍션 모듈과 메타데이터 기반의 모델을 결합한 클래스를 정의합니다. |
| MoEBlock.py | [RTIOD\WorkStation_MoE_3Channel\MMMMoE\MoEBlock.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE_3Channel/MMMMoE/MoEBlock.py) | 이 파일은 복잡한 머신 러닝 모델의 일부분을 구현한 PyTorch 클래스입니다. |
| RPN_ROI_Head.py | [RTIOD\WorkStation_MoE_3Channel\MMMMoE\RPN_ROI_Head.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE_3Channel/MMMMoE/RPN_ROI_Head.py) | 이 파일은 Faster R-CNN 모델의 RPN과 ROI Head 구성요소를 구축하는 함수를 정의합니다. |
| IRJsonDataset.py | [RTIOD\WorkStation_MoE_3Channel\IRJsonDataset.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE_3Channel/IRJsonDataset.py) | 이 파일은 JSON 형식의 데이터셋을 처리하는 PyTorch의 Dataset 클래스를 정의합니다. |
| IRJsonDataset_Augment.py | [RTIOD\WorkStation_MoE_3Channel\IRJsonDataset_Augment.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE_3Channel/IRJsonDataset_Augment.py) | 이 코드는 인프레이트(IR) JSON 데이터셋 클래스를 정의하며, JSON 파일에서 이미지와 객체 정보를 로드하고 데이터 증강을 적용합니다. |
| main.py | [RTIOD\WorkStation_MoE_3Channel\main.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE_3Channel/main.py) | 코드는 객체 검출 모델을 훈련시키는 프로그램으로, 데이터셋 로딩, 모델 정의, 학습 단계, 평가 등 주요 과정을 포함합니다. |
| Prediction_Json_Generator.py | [RTIOD\WorkStation_MoE_3Channel\Prediction_Json_Generator.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE_3Channel/Prediction_Json_Generator.py) | 이 파일은 텐서플로우와 파이토치를 사용해 객체 감지 모델을 평가하고 JSON 형식의 예측 파일을 생성하는 코드를 포함합니다. |
| Test_Step.py | [RTIOD\WorkStation_MoE_3Channel\Test_Step.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE_3Channel/Test_Step.py) | 이 파일은 텐서플로우를 사용해 모델의 단계별 테스트를 구현합니다. |
| Train_Step.py | [RTIOD\WorkStation_MoE_3Channel\Train_Step.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE_3Channel/Train_Step.py) | 이 파일은 딥러닝 모델의 단계별 학습 과정을 구현합니다. |
| Utils.py | [RTIOD\WorkStation_MoE_3Channel\Utils.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE_3Channel/Utils.py) | 이 파일은 객체 검출 모델의 성능을 평가하는 함수들을 포함하고 있으며, 박스 IOU 계산과 mAP 계산을 위한 함수들이 정의되어 있습니다. |
| Validation.py | [RTIOD\WorkStation_MoE_3Channel\Validation.py](file:///C:/junha/Git/RTIOD/WorkStation_MoE_3Channel/Validation.py) | 이 파일은 모델의 검증을 위한 함수를 정의하고 실행합니다. |

## WorkStation_SupCon

| File | Path | Description |
| :--- | :--- | :--- |
| Image_Crop_and_Folder.py | [RTIOD\WorkStation_SupCon\Preprocess\Image_Crop_and_Folder.py](file:///C:/junha/Git/RTIOD/WorkStation_SupCon/Preprocess/Image_Crop_and_Folder.py) | 이 코드는 JSON 파일에서 이미지 정보를 읽어 오버레이 정보를 기반으로 이미지를 자르고 저장합니다. |
| Batch_Sampler.py | [RTIOD\WorkStation_SupCon\Batch_Sampler.py](file:///C:/junha/Git/RTIOD/WorkStation_SupCon/Batch_Sampler.py) | 이 파일은 불균형 배치 샘플러를 구현하여 각 클래스에 대해 균일하게 배치를 샘플링하는 기능을 제공합니다. |
| main.py | [RTIOD\WorkStation_SupCon\main.py](file:///C:/junha/Git/RTIOD/WorkStation_SupCon/main.py) | 이 파일은SupContrast 알고리즘을 사용하여 이미지 분류 모델을 학습시키는 주요 함수와 데이터 로드, 손실 함수, 옵티마이저 설정을 포함합니다. |
| Model.py | [RTIOD\WorkStation_SupCon\Model.py](file:///C:/junha/Git/RTIOD/WorkStation_SupCon/Model.py) | 이 파일은 Resnet50을 기반으로 한 SupCon 모델을 정의하고 구현합니다. |
| SupContrast.py | [RTIOD\WorkStation_SupCon\SupContrast.py](file:///C:/junha/Git/RTIOD/WorkStation_SupCon/SupContrast.py) | SupContrast.py 파일은 감독된 상호 배제 손실 함수를 정의하며, 이는 비감독된 상호 배제 손실로도 사용될 수 있습니다. |
| Test_Step.py | [RTIOD\WorkStation_SupCon\Test_Step.py](file:///C:/junha/Git/RTIOD/WorkStation_SupCon/Test_Step.py) | 이 파일은 텐서플로우를 사용해 모델의 단계 테스트를 구현합니다. |
| Train_Step.py | [RTIOD\WorkStation_SupCon\Train_Step.py](file:///C:/junha/Git/RTIOD/WorkStation_SupCon/Train_Step.py) | 이 파일은 PyTorch를 사용하여 모델의 단계별 훈련을 구현합니다. |
| visualize.py | [RTIOD\WorkStation_SupCon\visualize.py](file:///C:/junha/Git/RTIOD/WorkStation_SupCon/visualize.py) | 이 파일은 PyTorch와 scikit-learn을 사용하여 이미지 데이터셋의 특징을 분류하고 T-SNE를 통해 2D 공간으로 시각화하는 코드입니다. |

