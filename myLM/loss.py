import torch
import torch.nn.functional as F

def clip_loss(image_feat, text_feat, logit_scale):
    """
    image_feat: [batch_size, feat_dim]
    text_feat: [batch_size, feat_dim]
    logit_scale: scalar, logit_scale = 1 / τ
    """
    # normalize features
    image_feat = F.normalize(image_feat, dim=-1)
    text_feat = F.normalize(text_feat, dim=-1)

    # logits
    logits_per_image = logit_scale * image_feat @ text_feat.t()
    logits_per_text = logits_per_image.t()

    # labels
    gt = torch.arange(len(image_feat), device=image_feat.device)

    # loss
    loss_i2t = F.cross_entropy(logits_per_image, gt)
    loss_t2i = F.cross_entropy(logits_per_text, gt)
    loss = (loss_i2t + loss_t2i) / 2
    return loss

if __name__ == "__main__":
    image_feat = torch.randn(10, 512)
    text_feat = torch.randn(10, 512)
    logit_scale = 100
    loss = clip_loss(image_feat, text_feat, logit_scale)
    print(loss)
