import imageio
import numpy as np
import skimage
import skimage.io
# import torch


# def hwc2chw(image):
#     return image.transpose(2, 1).transpose(0, 1)
# 
# 
# def chw2hwc(image):
#     return image.transpose(0, 1).transpose(2, 1)


def imwrite(path, image, gamma):
    if path.suffix == ".exr":
        imageio.imwrite(path, image.astype(np.float32))
    elif path.suffix == ".png":
        image_clip = np.clip(image, 0.0, 1.0)
        if image.shape[-1] == 4:
            image_bits = (np.power(image_clip[..., :3], 1 / gamma) * 255).astype(np.uint8)
            alpha_bits = (image_clip[..., -1:] * 255).astype(np.uint8)
            image_bits = np.concatenate([image_bits, alpha_bits], axis=-1)
        else:
            image_bits = (np.power(image_clip, 1 / gamma) * 255).astype(np.uint8)
        skimage.io.imsave(path, image_bits)
    else:
        raise NotImplementedError()


# def write_video(frames, fps, path):
#     # frames      - (t, c, h, w) float tensor
#     h, w = frames.size(-2), frames.size(-1)
#     frames = (frames * 255).to(torch.uint8)
#     out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (h, w), False)
#     for idx in range(frames.size(0)):
#         out.write(chw2hwc(frames[idx]).numpy())
#     out.release()


# def load_image(path):
#     image = imageio.imread(path)
#     if not (image.ndim == 2 or image.ndim == 3):
#         raise Exception("Invalid PNG Image num dimensions")
#     elif image.ndim == 2:
#         image = np.repeat(image[..., np.newaxis], 3, axis=-1)
#     else:
#         if image.shape[-1] == 4:
#             image = image[..., :3]
#     image = torch.tensor(image, dtype=torch.float32) / 255.0
#     image = hwc2chw(image)
#     return image
# 
# 
# def to_event_frames(images, threshold=0.1):
#     # image       - (t, h, w) float tensor
#     buffer = images[0, ...].clone()
#     spikes = [
#         torch.stack([
#             torch.zeros(images[0, ...].size()),
#             torch.ones(images[0, ...].size()),
#             torch.zeros(images[0, ...].size()),
#         ])
#     ]
#     for i in range(1, images.size(0)):
#         pos = (images[i, ...] - buffer) > threshold
#         neg = (images[i, ...] - buffer) < -threshold
#         # TODO: zero is redundant. remove later
#         zero = ~(pos + neg)
#         spikes.append(torch.stack([pos, zero, neg]))
#         buffer[pos] = images[i][pos].clone()
#         buffer[neg] = images[i][neg].clone()
#     return torch.stack(spikes)
# 
# 
# def serialize_event_frames(event_images, tstep=0.1):
#     # event_images- (t, p, h, w) tensor. p is polarity dimension
#     # tstep       - timestep size between event frames in seconds
#     events = []
#     for t in range(event_images.size(0)):
#         time = 1473347517.0 + t * tstep
#         pos_idx = torch.nonzero(event_images[t, 0, ...])
#         pos = [f"{time} {p[1]} {p[0]} 1" for p in pos_idx]
#         events.extend(pos)
#         neg_idx = torch.nonzero(event_images[t, 2, ...])
#         neg = [f"{time} {p[1]} {p[0]} 0" for p in neg_idx]
#         events.extend(neg)
#     return "{} {}\n{}".format(
#         event_images.size(2),
#         event_images.size(3),
#         '\n'.join(events),
#     )
