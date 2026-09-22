import torch
from torch import nn
from utils import setup_seed, get_fr_model, initialize_model, asr_calculation
import os
import json
from FaceParsing.interface import FaceParsing
from dataset import base_dataset
from torchvision import transforms
import torch.nn.functional as F
from torchvision.utils import save_image
import argparse
from torch.utils.data import Subset

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def save_json(path, data):
    """Replace atomically so a running result can be read safely."""
    temp_path = path + '.tmp'
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, allow_nan=False)
    os.replace(temp_path, path)


@torch.no_grad()
def main(args):
    setup_seed(args.seed)
    h = 512
    w = 512
    txt = ''
    ddim_steps = 45
    scale = 0
    classifier_scale = args.s
    batch_size = 2
    num_workers = 0

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    transform = transforms.Compose([transforms.Resize((512, 512)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    if args.dataset == 'celeba':
        dataset = base_dataset(dir='./celeba-hq_sample', transform=transform)
    elif args.dataset == 'ffhq':
        dataset = base_dataset(dir='./ffhq_sample', transform=transform)
    elif args.dataset == 'lfw':
        dataset = base_dataset(dir='./lfw_sample', transform=transform)
    elif args.dataset == 'cfp':
        dataset = base_dataset(dir='./cfp_sample', transform=transform)
    # dataset = base_dataset(dir='./s', transform=transform)
    if not 1 <= args.num <= len(dataset):
        raise ValueError(f"--num must be between 1 and {len(dataset)}")
    dataset = Subset(dataset, list(range(args.num)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    sampler = initialize_model('configs/stable-diffusion/v2-inpainting-inference.yaml',
                               'pretrained_model/512-inpainting-ema.ckpt')
    model = sampler.model

    # prng = np.random.RandomState(seed)
    # start_code = prng.randn(batch_size, 4, h // 8, w // 8)
    # start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)

    # attack_model_names = ['IR152', 'IRSE50', 'FaceNet', 'MobileFace']
    # attack_model_names = ['IRSE50']
    attack_model_names = [args.model]
    attack_model_dict = {'IR152': get_fr_model('IR152'), 'IRSE50': get_fr_model('IRSE50'),
                         'FaceNet': get_fr_model('FaceNet'), 'MobileFace': get_fr_model('MobileFace')}
    # attack_model_resize_dict = {'IR152': 112, 'IRSE50': 112, 'FaceNet': 160, 'MobileFace': 112}
    # cos_sim_scores_dict = {'IR152': [], 'IRSE50': [], 'FaceNet': [], 'MobileFace': []}
    # cos_sim_scores_dict = {'IRSE50': []}
    cos_sim_scores_dict = {args.model: []}
    clean_cos_sim_scores_dict = {args.model: []}

    for attack_model_name in attack_model_names:
        attack_model = attack_model_dict[attack_model_name]
        # Keep MobileFace black-box evaluation on CPU to save GPU memory.
        if attack_model_name == 'MobileFace':
            attack_model = attack_model.cpu()
            torch.cuda.empty_cache()
        eval_device = next(attack_model.parameters()).device
        classifier = {k: v for k, v in attack_model_dict.items() if k != attack_model_name}
        print('Black-box model:', attack_model_name)
        print('White-box models:', list(classifier))
        resize = nn.AdaptiveAvgPool2d((112, 112)) if attack_model_name != 'FaceNet' else nn.AdaptiveAvgPool2d(
            (160, 160))
        with torch.no_grad():
            for i, (image, tgt_image) in enumerate(dataloader):
                tgt_image = tgt_image.to(device)
                B = image.shape[0]

                face_parsing = FaceParsing()
                pred = face_parsing(image)
                image = image.to(device)

                # label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow',
                # 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
                def get_mask(number):
                    return pred == number

                masks = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
                # skin = get_mask(1)
                # nose = get_mask(2)
                # eye_gla = get_mask(3)
                # l_eye = get_mask(4)
                # r_eye = get_mask(5)
                # l_brow = get_mask(6)
                # r_brow = get_mask(7)
                # mouth = get_mask(10)
                # u_lip = get_mask(11)
                # l_lip = get_mask(12)
                mask = None
                for x in masks:
                    if mask is not None:
                        mask |= get_mask(x)
                    else:
                        mask = get_mask(x)
                mask = (mask == 0).float().reshape(B, 1, h, w).to(device)

                masked_image = image * (mask < 0.5)

                batch = {
                    "image": image.to(device),
                    "txt": B * [txt],
                    "mask": mask.to(device),
                    "masked_image": masked_image.to(device),
                }

                c = model.cond_stage_model.encode(batch["txt"])
                c_cat = list()
                for ck in model.concat_keys:
                    cc = batch[ck].float()
                    if ck != model.masked_image_key:
                        bchw = [B, 4, h // 8, w // 8]
                        cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                    else:
                        cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
                    c_cat.append(cc)
                c_cat = torch.cat(c_cat, dim=1)

                # cond
                cond = {"c_concat": [c_cat], "c_crossattn": [c]}

                # uncond cond
                uc_cross = model.get_unconditional_conditioning(B, "")
                uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

                shape = [model.channels, h // 8, w // 8]

                # start code
                _t = args.t  # 0-999
                z = model.get_first_stage_encoding(model.encode_first_stage(image.to(device)))
                t = torch.tensor([_t] * B, device=device)
                z_t = model.q_sample(x_start=z, t=t)

                samples_cfg, intermediates = sampler.sample(
                    ddim_steps,
                    B,
                    shape,
                    cond,
                    verbose=False,
                    eta=1.0,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc_full,
                    x_T=z_t,
                    _t=_t + 1,
                    log_every_t=1,
                    classifier=classifier,
                    classifier_scale=classifier_scale,
                    x_target=tgt_image
                )

                x_samples_ddim = model.decode_first_stage(samples_cfg)
                result = torch.clamp(x_samples_ddim, min=-1, max=1)
                # print(len(intermediates['x_inter']))
                # for i, x_inter in enumerate(intermediates['x_inter']):
                #  x_inter = torch.clamp(model.decode_first_stage(x_inter), min=-1, max=1)
                # for y, x in enumerate(range(x_inter.shape[0])):
                #    save_image((x_inter[x] + 1) / 2, f'inter/{i}_{y}.png')

                os.makedirs(os.path.join(args.save, 'img'), exist_ok=True)
                os.makedirs(os.path.join(args.save, 'msk'), exist_ok=True)
                print(i, batch_size)
                for x in range(result.shape[0]):
                    save_image((result[x] + 1) / 2, os.path.join(args.save, 'img', f'{i * batch_size + x}.png'))
                    save_image((masked_image[x] + 1) / 2, os.path.join(args.save, 'msk', f'{i * batch_size + x}_m.png'))

                # save_image((x_inter + 1) / 2, f'res/{i}_inter.png')

                # attack_model = attack_model_dict[attack_model_name]
                feature1 = attack_model(resize(result).to(eval_device)).reshape(B, -1)
                feature2 = attack_model(resize(tgt_image).to(eval_device)).reshape(B, -1)

                score = F.cosine_similarity(feature1, feature2)
                feature_src = attack_model(resize(image).to(eval_device)).reshape(B, -1)
                score_src = F.cosine_similarity(feature_src, feature2)
                print(f'[{attack_model_name}] SRC -> TGT:', score_src)
                print(f'[{attack_model_name}] ADV -> TGT:', score)
                clean_cos_sim_scores_dict[attack_model_name] += score_src.cpu().tolist()
                cos_sim_scores_dict[attack_model_name] += score.cpu().tolist()

                print('\n========== CLEAN ASR ==========')
                clean_asr_result = asr_calculation(clean_cos_sim_scores_dict)
                print('\n========== ATTACK ASR ==========')
                asr_result = asr_calculation(cos_sim_scores_dict)
                completed = len(cos_sim_scores_dict[attack_model_name])
                save_json(os.path.join(args.save, 'result.json'), asr_result)
                save_json(os.path.join(args.save, 'clean_result.json'), clean_asr_result)
                save_json(os.path.join(args.save, 'progress.json'), {
                    'completed': completed,
                    'total': len(dataset),
                    'finished': completed == len(dataset),
                    'dataset': args.dataset,
                    'blackbox_model': attack_model_name,
                    'whitebox_models': list(classifier),
                    'seed': args.seed,
                    't': args.t,
                    'classifier_scale': args.s,
                    'clean_scores': clean_cos_sim_scores_dict,
                    'attack_scores': cos_sim_scores_dict,
                })
                print(f'Completed {completed}/{len(dataset)}; updated {args.save}/result.json', flush=True)

                # feature3 = attack_model(resize(x_inter)).reshape(B, -1)
                # score = F.cosine_similarity(feature3, feature2)
                # print(score)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['IR152', 'IRSE50', 'FaceNet', 'MobileFace'], default='IR152')
    parser.add_argument('--dataset', choices=['celeba', 'ffhq', 'lfw', 'cfp'], default='celeba')
    parser.add_argument('--num', type=int, default=6)
    parser.add_argument('--t', type=int, default=50)
    parser.add_argument('--save', type=str, default='res')
    parser.add_argument('--s', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    main(args)
