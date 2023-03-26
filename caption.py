from PIL import Image
from torchvision import transforms
from transformers import OFATokenizer, OFAModel, ViTFeatureExtractor, ViTForImageClassification
from generate import sequence_generator
import torch
import argparse
import matplotlib.pyplot as plt
from captioning_dataset import ImageFolderDataset, ImageListDatasetFromGradio
from torch.utils.data import Dataset, DataLoader



def captioning(*args, arguments):

    # Caption generation

    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    resolution = 384
    patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(), 
            transforms.Normalize(mean=mean, std=std)
        ])

    tokenizer = OFATokenizer.from_pretrained(arguments.cap_dir)

    txt = " what does the image describe?"
    cap_inputs = tokenizer([txt], return_tensors="pt").input_ids
    # folder_path=args.folder_path
    img_list=[]
    for img in args:
        img_list.append(img)

    dataset=ImageListDatasetFromGradio(img_list, transform=patch_resize_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # patch_img = patch_resize_transform(img).unsqueeze(0)

    # Captioning model definition
    cap_model = OFAModel.from_pretrained('OFA-base', use_cache=True)
    generator = sequence_generator.SequenceGenerator(
                        tokenizer=tokenizer,
                        beam_size=5,
                        max_len_b=16, 
                        min_len=0,
                        no_repeat_ngram_size=3,
                    )

    annot_path=(arguments.annot_path +'/'+ arguments.folder_path.split('/')[1]+'.txt')
    print(annot_path)
    annot=open(annot_path, 'w')
    cls_list=[]
    cap_list=[]
    corpus=""
    for batch_idx, image in enumerate(dataloader):    
        data = {}
        data["net_input"] = {"input_ids": cap_inputs, 'patch_images': image, 'patch_masks':torch.tensor([True])}
        gen_output = generator.generate([cap_model], data)
        gen = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]

        # using the generator of huggingface version
        caption=tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()

        annot.write("\nCaption\n{}".format(caption)) #

        print("Caption")
        print(caption)
        cap_list.append(caption)


    print(cap_list)
    annot.close()

    return cap_list
