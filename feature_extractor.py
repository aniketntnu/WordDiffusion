import torch.nn as nn
import timm
import torch
class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='resnet50', num_classes=0, pretrained=True, trainable=True
    ):
        super().__init__()
        # checkpoint_path='/home/hankyul/.cache/torch/hub/checkpoints/resnet50_a1_0-14fe96d1.pth'


        """
        self.model = timm.create_model(
            model_name, pretrained, num_classes=num_classes, global_pool="max"
        )

        """
        print("\n\t model_name:",model_name)
        #logger.info(" model_name:",model_name)
        
        self.model = timm.create_model(
            model_name, pretrained=False, num_classes=num_classes, global_pool="max"
        )
        """
        # Define the local checkpoint path
        checkpoint_path = "/cluster/datastore/aniketag/allData/wordStylist/writerStyle/diffPenWeights/iam_style_diffusionpen_triplet.pth"

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda:0"))  # Adjust for GPU if needed

        # Load the weights into the model
        self.model.load_state_dict(checkpoint)
        """
        #self.model = torch.compile(self.model, backend="inductor")
        for p in self.model.parameters():
            p.requires_grad = trainable
    def forward(self, x):
        x = self.model(x)
        return x   