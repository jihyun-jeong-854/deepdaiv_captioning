from PIL import Image
from torchvision import transforms
from transformers import OFATokenizer, OFAModel, ViTFeatureExtractor, ViTForImageClassification
from generate import sequence_generator
import torch
import argparse
import matplotlib.pyplot as plt
from cap_cls_dataset import ImageFolderDataset, ImageListDatasetFromGradio
from torch.utils.data import Dataset, DataLoader



def cap_cls(*args, arguments):

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
    # Cls model definition
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')    

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
        cap_model = OFAModel.from_pretrained('OFA-base', use_cache=False)
        gen = cap_model.generate(cap_inputs, patch_images=image, num_beams=5, no_repeat_ngram_size=3, max_length=50) 
        caption=tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()

        image=image.squeeze()
        cls_inputs = feature_extractor(images=image, return_tensors="pt")
        cls_outputs = model(**cls_inputs)
        logits = cls_outputs.logits
        annot.write("\nCaption\n{}".format(caption)) #
        annot.write("\nCls\n")
        print("=========================================================================")
        print("Cls")
        if arguments.pred_num==1:
            top1_predicted_class_idx = logits.argmax(-1).item()
            top1_predicted_class= model.config.id2label[top1_predicted_class_idx]
            predicted_class=top1_predicted_class
            predicted_prob=round((torch.exp(torch.max(logits)) / torch.sum(torch.exp(logits.squeeze()))).item()*100)
            print("Predicted class: {}, {}\n".format(predicted_class, predicted_prob))

        else:
            topN_predicted_class_idx=logits.argsort(descending=True)[:, :int(arguments.pred_num)].squeeze()
            predicted_prob=[round(i) for i in ((torch.exp(logits.squeeze()[topN_predicted_class_idx])\
                            /torch.sum(torch.exp(logits.squeeze())))*100).tolist()]
            topN_predicted_class=[model.config.id2label[int(i)] for i in topN_predicted_class_idx]   
            predicted_class=topN_predicted_class 
            for i in range(int(arguments.pred_num)):
                print("{}. {}, {}%".format(i+1, predicted_class[i], predicted_prob[i]))
                annot.write("{}. {}, {}%\n".format(i+1, predicted_class[i], predicted_prob[i]))

        print("Caption")
        print(caption)
        cap_list.append(caption)
        # 모델별로
        # predicted_class: 클래스 리스트
        # predicted_prob: 확률 리스트
        # caption: 캡션

    print(cap_list)
    annot.close()

    return cap_list
