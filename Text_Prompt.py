import torch
import clip
import numpy as np


label_text_map = []
with open('text/ntu120_label_map.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        label_text_map.append(line.rstrip().lstrip())

ntu_semantic_text_map2 = []
with open('text/ntu_pasta_openai_t01.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(';')
        ntu_semantic_text_map2.append(temp_list)

ntu_semantic_text_map_gpt35= []
with open('text/ntu_gpt_35.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(';')
        ntu_semantic_text_map_gpt35.append(temp_list)

ntu_semantic_text_map_gpt35_4part= []
with open('text/ntu60_gpt_35_4part.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(';')
        ntu_semantic_text_map_gpt35_4part.append(temp_list)

ntu_semantic_text_map_gpt35_2part= []
with open('text/ntu60_gpt_35_2part.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(';')
        ntu_semantic_text_map_gpt35_2part.append(temp_list)

pkuv1_semantic_text_map_gpt35_2part= []
with open('text/pkuv1_gpt_35_2part.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(';')
        pkuv1_semantic_text_map_gpt35_2part.append(temp_list)

pkuv1_semantic_text_map_gpt35_4part= []
with open('text/pkuv1_gpt_35_4part.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(';')
        pkuv1_semantic_text_map_gpt35_4part.append(temp_list)

pkuv1_semantic_text_map_gpt35= []
with open('text/pkuv1_gpt_35.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(';')
        pkuv1_semantic_text_map_gpt35.append(temp_list)

ntu_spatial_attribute_text_map = []
with open('text/ntu_spatial_attributes.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        # temp_list = line.rstrip().lstrip().split(';')
        ntu_spatial_attribute_text_map.append(line)

ntu_spatial_temporal_attribute_text_map = []
with open('text/ntu_spatial_temporal_attributes.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        # temp_list = line.rstrip().lstrip().split(';')
        ntu_spatial_temporal_attribute_text_map.append(line)

# load clip model
def ntu_attributes(device):
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load('ViT-L/14@336px', device)
    clip_model.cuda(device)
    ntu120_semantic_feature_dict = {}
    ntu_spatial_attribute_feature_dict = {}
    ntu_spatial_temporal_attribute_feature_dict = {}
    with torch.no_grad():
        # class semantic vector
        # for idx, class_semantic_label in enumerate(ntu_semantic_text_map2):
        text_dict = {}
        num_text_aug = 5   # 7
        for ii in range(num_text_aug):
            if ii == 0:
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[ii])) for pasta_list in ntu_semantic_text_map2])   # class
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[ii])) for pasta_list in ntu_semantic_text_map_gpt35])   # class
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[ii])) for pasta_list in pkuv1_semantic_text_map_gpt35])   # class
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[ii])) for pasta_list in ntu_semantic_text_map_gpt35_4part])   # class
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[ii])) for pasta_list in ntu_semantic_text_map_gpt35_2part])   # class
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[ii])) for pasta_list in pkuv1_semantic_text_map_gpt35_2part])   # class
                text_dict[ii] = torch.cat([clip.tokenize((pasta_list[ii])) for pasta_list in pkuv1_semantic_text_map_gpt35_4part])   # class
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0]+','+pasta_list[1]+','+pasta_list[2]+','+pasta_list[3]+','+pasta_list[4]+','+pasta_list[5]+','+pasta_list[6])) for pasta_list in ntu_semantic_text_map2])   # class
            elif ii == 1:
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[1])) for pasta_list in ntu_semantic_text_map2])   # class, head
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[1])) for pasta_list in ntu_semantic_text_map_gpt35])
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[1])) for pasta_list in pkuv1_semantic_text_map_gpt35])
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[1])) for pasta_list in ntu_semantic_text_map_gpt35_4part])
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[1]),context_length=77, truncate=True) for pasta_list in ntu_semantic_text_map_gpt35_2part])
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[1]),context_length=77, truncate=True) for pasta_list in pkuv1_semantic_text_map_gpt35_2part])
                text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[1]),context_length=77, truncate=True) for pasta_list in pkuv1_semantic_text_map_gpt35_4part])
            elif ii == 2:
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] +','+ pasta_list[2])) for pasta_list in paste_text_map2])  # class hand 
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[2])) for pasta_list in ntu_semantic_text_map2])  # class hand 
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[2])) for pasta_list in ntu_semantic_text_map_gpt35])
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[2])) for pasta_list in pkuv1_semantic_text_map_gpt35])
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[2])) for pasta_list in ntu_semantic_text_map_gpt35_4part])
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[2]),context_length=77, truncate=True) for pasta_list in ntu_semantic_text_map_gpt35_2part])
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[2]),context_length=77, truncate=True) for pasta_list in pkuv1_semantic_text_map_gpt35_2part])
                text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[2]),context_length=77, truncate=True) for pasta_list in pkuv1_semantic_text_map_gpt35_4part])
            elif ii == 3:
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] +','+ pasta_list[3])) for pasta_list in paste_text_map2])  # class arm
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[3])) for pasta_list in ntu_semantic_text_map2])  # class arm
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[3])) for pasta_list in ntu_semantic_text_map_gpt35])
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[3])) for pasta_list in pkuv1_semantic_text_map_gpt35])
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[3])) for pasta_list in ntu_semantic_text_map_gpt35_4part])
                text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[3])) for pasta_list in pkuv1_semantic_text_map_gpt35_4part])
            elif ii == 4:
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] +','+ pasta_list[4])) for pasta_list in paste_text_map2])  # class hip
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[4])) for pasta_list in ntu_semantic_text_map2])  # class hip
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[4])) for pasta_list in ntu_semantic_text_map_gpt35])
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[4])) for pasta_list in pkuv1_semantic_text_map_gpt35])
                # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[4])) for pasta_list in ntu_semantic_text_map_gpt35_4part])
                text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[4])) for pasta_list in pkuv1_semantic_text_map_gpt35_4part])
            # elif ii == 5:
            #     # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] +','+ pasta_list[5])) for pasta_list in paste_text_map2])  # class leg
            #     # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[5])) for pasta_list in ntu_semantic_text_map2])  # class leg
            #     # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[5])) for pasta_list in ntu_semantic_text_map_gpt35])
            #     text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[5])) for pasta_list in pkuv1_semantic_text_map_gpt35])
            # else:
            #     # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] +','+ pasta_list[6])) for pasta_list in paste_text_map2])  # class foot
            #     # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[6])) for pasta_list in ntu_semantic_text_map2])  # class foot
            #     # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[6])) for pasta_list in ntu_semantic_text_map_gpt35])
            #     text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[6])) for pasta_list in pkuv1_semantic_text_map_gpt35])
            ntu120_semantic_feature_dict[ii] = clip_model.float().encode_text(text_dict[ii].to(device))
        # spatial attribute
        for part_name in ntu_spatial_attribute_text_map:
            ntu_spatial_attribute_feature_dict[part_name] = clip_model.float().encode_text(clip.tokenize(part_name).to(device))
        # spatial temporal attribute
        tokenize_tmp = torch.cat([clip.tokenize(ntu_spatial_temporal_attribute_text_map[i],truncate=True) for i in range(0, 4)]).to(device)
        ntu_spatial_temporal_attribute_feature_dict['head'] = clip_model.float().encode_text(tokenize_tmp)
        tokenize_tmp = torch.cat([clip.tokenize(ntu_spatial_temporal_attribute_text_map[i],truncate=True) for i in range(4, 10)]).to(device)
        ntu_spatial_temporal_attribute_feature_dict['hand'] = clip_model.float().encode_text(tokenize_tmp)
        tokenize_tmp = torch.cat([clip.tokenize(ntu_spatial_temporal_attribute_text_map[i],truncate=True) for i in range(10, 15)]).to(device)
        ntu_spatial_temporal_attribute_feature_dict['arm'] = clip_model.float().encode_text(tokenize_tmp)
        tokenize_tmp = torch.cat([clip.tokenize(ntu_spatial_temporal_attribute_text_map[i],truncate=True) for i in range(15, 19)]).to(device)
        ntu_spatial_temporal_attribute_feature_dict['hip'] = clip_model.float().encode_text(tokenize_tmp)
        tokenize_tmp = torch.cat([clip.tokenize(ntu_spatial_temporal_attribute_text_map[i],truncate=True) for i in range(19, 23)]).to(device)
        ntu_spatial_temporal_attribute_feature_dict['leg'] = clip_model.float().encode_text(tokenize_tmp)
        tokenize_tmp = torch.cat([clip.tokenize(ntu_spatial_temporal_attribute_text_map[i],truncate=True) for i in range(23, 27)]).to(device)
        ntu_spatial_temporal_attribute_feature_dict['foot'] = clip_model.float().encode_text(tokenize_tmp)
    
    torch.save(ntu120_semantic_feature_dict,'/DATA3/cy/STAR/data/text_feature/pkuv1_semantic_feature_dict_gpt35_4part.tar')
    torch.save(ntu_spatial_attribute_feature_dict,'/DATA3/cy/STAR/data/text_feature/pkuv1_spatial_attribute_feature_dict_gpt35.tar')
    torch.save(ntu_spatial_temporal_attribute_feature_dict,'/DATA3/cy/STAR/data/text_feature/pkuv1_spatial_temporal_attribute_feature_dict_gpt35.tar')
    return ntu120_semantic_feature_dict, ntu_spatial_attribute_feature_dict, ntu_spatial_temporal_attribute_feature_dict



    # caption_tokenize = torch.cat([clip.tokenize((caption.strip()), truncate=True) for caption in captions]).to(device)
    # # print('CLIP Tokenize Processing Finish!')
    # caption_feature = clip_model.float().encode_text(caption_tokenize)
    # # print('Extracting CLIP Text Feature Finish!')
    # caption_feature_list.append(caption_feature.cpu().numpy())











# def text_prompt():
#     text_aug = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
#                 f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
#                 f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
#                 f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
#                 f"The man is {{}}", f"The woman is {{}}"]
#     text_dict = {}
#     num_text_aug = len(text_aug)

#     for ii, txt in enumerate(text_aug):
#         text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for c in label_text_map])


#     classes = torch.cat([v for k, v in text_dict.items()])

#     return classes, num_text_aug,text_dict




# def text_prompt_openai_pasta_pool_4part():
#     print("Use text prompt openai pasta pool")
#     text_dict = {}
#     num_text_aug = 7

#     for ii in range(num_text_aug):
#         if ii == 0:
#             text_dict[ii] = torch.cat([clip.tokenize((pasta_list[ii])) for pasta_list in paste_text_map2])   # class
#         elif ii == 1:
#             text_dict[ii] = torch.cat([clip.tokenize((','.join(pasta_list[1:2]))) for pasta_list in paste_text_map2])   # class, head
#         elif ii == 2:
#             # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] +','+ pasta_list[2])) for pasta_list in paste_text_map2])  # class hand 
#             text_dict[ii] = torch.cat([clip.tokenize((','+ pasta_list[2])) for pasta_list in paste_text_map2])  # class hand 
#         elif ii == 3:
#             # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] +','+ pasta_list[3])) for pasta_list in paste_text_map2])  # class arm
#             text_dict[ii] = torch.cat([clip.tokenize((','+ pasta_list[3])) for pasta_list in paste_text_map2])  # class arm
#         elif ii == 4:
#             # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] +','+ pasta_list[4])) for pasta_list in paste_text_map2])  # class hip
#             text_dict[ii] = torch.cat([clip.tokenize((','+ pasta_list[4])) for pasta_list in paste_text_map2])  # class hip
#         elif ii == 5:
#             # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] +','+ pasta_list[5])) for pasta_list in paste_text_map2])  # class leg
#             text_dict[ii] = torch.cat([clip.tokenize((','+ pasta_list[5])) for pasta_list in paste_text_map2])  # class leg
#         else:
#             # text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] +','+ pasta_list[6])) for pasta_list in paste_text_map2])  # class foot
#             text_dict[ii] = torch.cat([clip.tokenize((','+ pasta_list[6])) for pasta_list in paste_text_map2])  # class foot


#     classes = torch.cat([v for k, v in text_dict.items()])
    
#     return classes, num_text_aug, text_dict

