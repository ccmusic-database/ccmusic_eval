import os
import torch
import torch.nn as nn
import torchvision.models as models
from modelscope.msdatasets import MsDataset
from utils import download

TRAIN_MODES = ["no_pretrain", "linear_probe", "full_finetune"]


class WCE(nn.CrossEntropyLoss):
    def __init__(self, sample_sizes: list):
        super(WCE, self).__init__()
        weights = torch.tensor(
            [1.0 / size for size in sample_sizes], dtype=torch.float32
        )
        self.weight = weights / weights.sum()


class Net:
    def __init__(
        self,
        backbone: str,
        cls_num: int,
        train_mode: int,
        imgnet_ver="v1",
        weight_path="",
    ):
        if not train_mode in range(len(TRAIN_MODES)):
            raise ValueError(f"Unsupported training mode {train_mode}.")

        if not hasattr(models, backbone):
            raise ValueError(f"Unsupported model {backbone}.")

        self.imgnet_ver = imgnet_ver
        self.output_size = 512
        self.training = weight_path == ""
        self.full_finetune = train_mode != 1
        self.type, self.weight_url, self.input_size = self._model_info(backbone)
        self.model: torch.nn.Module = eval("models.%s()" % backbone)
        linear_output = self._set_outsize()
        if self.training:
            if train_mode > 0:
                weight_path = self._download_model(self.weight_url)
                checkpoint = (
                    torch.load(weight_path)
                    if torch.cuda.is_available()
                    else torch.load(weight_path, map_location="cpu")
                )
                self.model.load_state_dict(checkpoint, False)

            for parma in self.model.parameters():
                parma.requires_grad = self.full_finetune

            self._set_classifier(cls_num, linear_output)
            self.model.train()

        else:
            self._set_classifier(cls_num, linear_output)
            checkpoint = (
                torch.load(weight_path)
                if torch.cuda.is_available()
                else torch.load(weight_path, map_location="cpu")
            )
            self.model.load_state_dict(checkpoint, False)
            self.model.eval()

    def _get_backbone(self, backbone_ver, backbone_list):
        for backbone_info in backbone_list:
            if backbone_ver == backbone_info["ver"]:
                return backbone_info

        raise ValueError("[Backbone not found] Please check if --model is correct!")

    def _model_info(self, backbone: str):
        backbone_list = MsDataset.load(
            "monetjoe/cv_backbones",
            split=self.imgnet_ver,
            cache_dir="./__pycache__",
            # download_mode="force_redownload",
        )
        backbone_info = self._get_backbone(backbone, backbone_list)
        return (
            str(backbone_info["type"]),
            str(backbone_info["url"]),
            int(backbone_info["input_size"]),
        )

    def _download_model(self, weight_url: str, model_dir="./__pycache__"):
        weight_path = f'{model_dir}/{weight_url.split("/")[-1]}'
        os.makedirs(model_dir, exist_ok=True)
        if not os.path.exists(weight_path):
            download(weight_url, weight_path)

        return weight_path

    def _create_classifier(self, cls_num: int, linear_output: bool):
        q = (1.0 * self.output_size / cls_num) ** 0.25
        l1 = int(q * cls_num)
        l2 = int(q * l1)
        l3 = int(q * l2)
        if linear_output:
            return nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.output_size, l3),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(l3, l2),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(l2, l1),
                nn.ReLU(inplace=True),
                nn.Linear(l1, cls_num),
            )

        else:
            return nn.Sequential(
                nn.Dropout(),
                nn.Conv2d(self.output_size, l3, kernel_size=(1, 1), stride=(1, 1)),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten(),
                nn.Linear(l3, l2),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(l2, l1),
                nn.ReLU(inplace=True),
                nn.Linear(l1, cls_num),
            )

    def _set_outsize(self, debug_mode=False):
        for name, module in self.model.named_modules():
            if (
                str(name).__contains__("classifier")
                or str(name).__eq__("fc")
                or str(name).__contains__("head")
                or hasattr(module, "classifier")
            ):
                if isinstance(module, torch.nn.Linear):
                    self.output_size = module.in_features
                    if debug_mode:
                        print(
                            f"{name}(Linear): {self.output_size} -> {module.out_features}"
                        )

                    return True

                if isinstance(module, torch.nn.Conv2d):
                    self.output_size = module.in_channels
                    if debug_mode:
                        print(
                            f"{name}(Conv2d): {self.output_size} -> {module.out_channels}"
                        )

                    return False

        return False

    def _set_classifier(self, cls_num, linear_output):
        if self.type == "convnext":
            del self.model.classifier[2]
            self.model.classifier = nn.Sequential(
                *list(self.model.classifier)
                + list(self._create_classifier(cls_num, linear_output))
            )
            self.classifier = self.model.classifier

        elif self.type == "maxvit":
            del self.model.classifier[5]
            self.model.classifier = nn.Sequential(
                *list(self.model.classifier)
                + list(self._create_classifier(cls_num, linear_output))
            )
            self.classifier = self.model.classifier

        elif hasattr(self.model, "classifier"):
            self.model.classifier = self._create_classifier(cls_num, linear_output)
            self.classifier = self.model.classifier

        elif hasattr(self.model, "fc"):
            self.model.fc = self._create_classifier(cls_num, linear_output)
            self.classifier = self.model.fc

        elif hasattr(self.model, "head"):
            self.model.head = self._create_classifier(cls_num, linear_output)
            self.classifier = self.model.head

        else:
            self.model.heads.head = self._create_classifier(cls_num, linear_output)
            self.classifier = self.model.heads.head

        for parma in self.classifier.parameters():
            parma.requires_grad = True

    def get_input_size(self):
        return self.input_size

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
            self.model = self.model.cuda()

        if self.type == "googlenet" and self.training:
            return self.model(x)[0]
        else:
            return self.model(x)

    def parameters(self):
        if self.full_finetune:
            return self.model.parameters()
        else:
            return self.classifier.parameters()

    def state_dict(self):
        return self.model.state_dict()
