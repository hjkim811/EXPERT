import os
import re
import json
import torch
import argparse
import numpy as np
import scipy.stats
from PIL import Image
from tqdm import tqdm
from transformers import TextStreamer

from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


def preprocess_dataset(base_fold, ann_file):
    with open(os.path.join(base_fold, ann_file), 'r') as f:
        data = json.load(f)

    img_list = []
    txt_list = []
    human_scores = []

    if ann_file in ['flickr8k.json', 'crowdflower_flickr8k.json']:
        for k, v in list(data.items()):
            for human_judgement in v['human_judgement']:
                if np.isnan(human_judgement['rating']):
                    print('NaN')
                    continue
                human_scores.append(human_judgement['rating'])
                img_list.append(human_judgement['image_path'])
                txt_list.append(' '.join(human_judgement['caption'].split()))
                
    elif ann_file in ['composite.json']:
        for key in data:
            for d in data[key]:
                human_scores.append(d['human'])
                img_list.append(d['image'])
                txt_list.append(' '.join(d['caption'].split()))

    elif ann_file in ['pascal_50s.json']:
        category_data = data[args.pascal_category]
        for d in category_data:
            img_list.append(d['image'])
            txt_list.append(d['captions'])
            human_scores.append(d['label'])

    elif ann_file in ['polaris.json', 'polaris_test.json']:
        for d in data:
            img_list.append(f"Polaris/{d['imgid']}.jpg")
            txt_list.append(d['mt'])
            human_scores.append(d['score'])

    elif ann_file in ['nebula.json', 'nebula_test.json']:
        for d in data:
            img_list.append(f"Nebula/{d['imgid']}.jpg")
            txt_list.append(d['mt'])
            human_scores.append(d['score'])

    else:
        raise Exception(f'Unexpected Input JSON File ({ann_file})')
    
    return img_list, txt_list, human_scores


def get_counts(human_labels, pred_labels):
    true_count = 0
    false_count = 0
    tie_count = 0
    
    for human, model in zip(human_labels, pred_labels):
        if model == -1:
            tie_count += 1
        elif human == model:
            true_count += 1
        else:
            false_count += 1

    return true_count, false_count, tie_count


def compute_metrics(results, ann_file):
    if ann_file != 'pascal_50s.json':
        expert_scores = [item['expert_score'] for item in results]
        human_scores = [item['human_score'] for item in results]

        print()
        print('[Results]')
        if ann_file == 'crowdflower_flickr8k.json':
            print(f"Correlation (Tau-b): {round(100*scipy.stats.kendalltau(expert_scores, human_scores, variant='b')[0], 3)}")
        else:
            print(f"Correlation (Tau-c): {round(100*scipy.stats.kendalltau(expert_scores, human_scores, variant='c')[0], 3)}")
                
    else:
        expert_pred_labels = [item['expert_pred_label'] for item in results]
        human_labels = [item['human_label'] for item in results]
        true, false, tie = get_counts(human_labels, expert_pred_labels)
        accuracy = round(true / (true + false + tie), 3)

        print()
        print(f'[Results]')
        print(f'Accuracy: {round(100*accuracy, 1)}')


def get_scoring_prompt(caption):
    prompt = f"Evaluate the caption and assign a score on a scale of 0.0 to 1.0.\n\nCaption: {caption}\n\nScore (0.0~1.0):"

    return prompt


def get_explanation_prompt(caption):
    prompt = f"Provide a brief explanation for the score based on three criteria: Fluency, Relevance, and Descriptiveness.\n\nCaption: {caption}\n\nEvaluation Criteria:\n- Fluency: Whether the caption is fluent, natural, and grammatically correct.\n- Relevance: Whether the sentence correctly describes the visual content and be closely relevant to the image.\n- Descriptiveness: Whether the sentence is a precise, informative caption that describes important details of the image.\n\nOutput Format:\nFluency: {{Provide explanation here.}}\nRelevance: {{Provide explanation here.}}\nDescriptiveness: {{Provide explanation here.}}"

    return prompt


def parse_explanation(outputs):
    try:
        regex = r"Fluency:\s*(.*?)\nRelevance:\s*(.*?)\nDescriptiveness:\s*(.*)"
        m = re.search(regex, outputs, re.DOTALL)
        if m:
            explanation_dict = {
                'fluency': m.group(1).strip(),
                'relevance': m.group(2).strip(),
                'descriptiveness': m.group(3).strip()
            }
        else:
            raise ValueError()
            
    except Exception as e:
        print(f'Parsing Error:\n{outputs}\n')
        explanation_dict = None

    return explanation_dict


def main(args):
    # Load and preprocess dataset
    img_list, txt_list, human_scores = preprocess_dataset(args.base_fold, args.input_json)
            
    device_map="auto"
    kwargs = {"device_map": device_map}
    kwargs['torch_dtype'] = torch.float16
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    rate2token = {s : tokenizer.encode(str(s))[-1] for s in range(10)}

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode


    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles
    
    os.makedirs(args.result_fold, exist_ok=True)
    results = []

    # Datasets other than Pascal-50S
    if args.input_json != 'pascal_50s.json':
        previous_image_file = None
        previous_caption = None
        previous_explanation = None
        previous_expert_score = None

        for idx, (image_file, caption, human_score) in enumerate(zip(tqdm(img_list), txt_list, human_scores)):

            # Skip inference for repeated image-caption pairs
            if image_file == previous_image_file and caption == previous_caption:
                item = {}
                item['image_file'] = previous_image_file
                item['caption'] = previous_caption
                item['expert_score'] = previous_expert_score
                if args.explanation:
                    item['explanation'] = previous_explanation
                item['human_score'] = human_score

                results.append(item)
                
                if args.print_logs:
                    print('\nIdentical image-caption pair. Model inference not running.')
                    print('-'*100)

            # Run inference for new image-caption pairs
            else:
                item = {}
                item['image_file'] = image_file
                item['caption'] = caption
                
                conv = conv_templates[conv_mode].copy()
                roles = conv.roles
                
                image = Image.open(os.path.join('images', image_file)).convert('RGB')
                image_tensor = process_images([image], image_processor, args)
                if type(image_tensor) is list:
                    image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
                else:
                    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

                if args.explanation:
                    prompts = [get_scoring_prompt(caption), get_explanation_prompt(caption)]
                else:
                    prompts = [get_scoring_prompt(caption)]

                score_smoothing = True

                for prompt in prompts:
                    if image is not None:
                        if model.config.mm_use_im_start_end:
                            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
                        else:
                            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
                        conv.append_message(conv.roles[0], prompt)
                        image = None
                    else:
                        conv.append_message(conv.roles[0], prompt)

                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    
                    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                    
                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                    if args.print_logs and score_smoothing is False:
                        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                    else:
                        streamer = None

                    with torch.inference_mode():
                        if args.print_logs:
                            if score_smoothing:
                                print(f'\nSCORING:')
                            else:
                                print(f'\nEXPLANATION:')

                        output_dict = model.generate(
                                    input_ids,
                                    images=image_tensor,
                                    do_sample=False, # Greedy decoding
                                    num_beams=1, # Greedy decoding
                                    max_new_tokens=512,
                                    streamer=streamer,
                                    use_cache=True,
                                    stopping_criteria=[stopping_criteria],
                                    output_scores=True,
                                    return_dict_in_generate=True,
                                )
                    output_ids = output_dict.sequences
                    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

                    # 1st turn (Scoring stage)
                    if score_smoothing:
                        dotsnumbersdots = re.sub(f'[^\d\.]', '', outputs)
                        numbersdots = re.sub(f'^\.+', '', dotsnumbersdots)
                        numbers = re.sub(r'\.+$', '', numbersdots)
                        score_check = float(numbers)

                        if score_check < 0 or score_check > 1:
                            continue
                        
                        elif score_check < 1.0:
                            num_index_in_score = str(score_check).index('.') + 1
                            find_num = int(str(score_check)[num_index_in_score])
                            num_index_in_token = (output_ids[0,1:] == rate2token[find_num]).nonzero().squeeze()
                            
                            if len(num_index_in_token.shape) > 0:   
                                if find_num == 0:   
                                    num_index_in_token = num_index_in_token[1] 
                                else:       
                                    num_index_in_token = num_index_in_token[0]
                            probs = output_dict.scores[num_index_in_token]
                            probs = torch.nn.functional.softmax(probs, dim=-1)[0]
                            
                            score = 0.
                            for rate, token in rate2token.items():
                                score += probs[token] * rate * 0.1

                            if len(str(score_check)) > 3:
                                num2_index_in_score = str(score_check).index('.') + 2
                                find_num2 = int(str(score_check)[num2_index_in_score])
                                num2_index_in_token = (output_ids[0,1:] == rate2token[find_num2]).nonzero().squeeze()
                                if len(num2_index_in_token.shape) > 0:
                                    num2_index_in_token = num2_index_in_token[1]
                                probs2 = output_dict.scores[num2_index_in_token]
                                probs2 = torch.nn.functional.softmax(probs2, dim=-1)[0]
                            
                                for rate, token in rate2token.items():
                                    score += probs2[token] * rate * 0.01
                        
                        else:
                            num_index_in_token = (output_ids[0,1:] == rate2token[1]).nonzero().squeeze()
                            probs = output_dict.scores[num_index_in_token]
                            probs = torch.nn.functional.softmax(probs, dim=-1)[0]
                            
                            score = 0.9 * probs[rate2token[0]] + probs[rate2token[1]]
                        
                        conv.messages[-1][-1] = str(score.cpu().item()) + '</s>'
                        if args.print_logs:
                            print('Score: ', score.item())

                        item['expert_score'] = score.item()
                        previous_expert_score = score.item()

                        score_smoothing = False
                    
                    # 2nd turn (Explanation stage)
                    else:
                        conv.messages[-1][-1] = outputs
                        item['explanation'] = parse_explanation(outputs)
                        previous_explanation = parse_explanation(outputs)

                item['human_score'] = human_score
                results.append(item)
                previous_image_file = image_file
                previous_caption = caption
                
                if args.print_logs:
                    print('-'*100)

        if args.explanation:
            result_file = f"{model_name}_exp_{args.input_json[:-5]}.json"
        else:
            result_file = f"{model_name}_{args.input_json[:-5]}.json"
            
        # Save results
        with open(os.path.join(args.result_fold, result_file), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    # Pascal-50S dataset
    else:
        for idx, (image_file, captions, human_score) in enumerate(zip(tqdm(img_list), txt_list, human_scores)):
            item = {}
            item['image_file'] = image_file

            for i in range(2):
                caption = captions[i]
                item[f'caption_{i+1}'] = caption
                
                conv = conv_templates[conv_mode].copy()
                roles = conv.roles
                
                image = Image.open(os.path.join('images', image_file)).convert('RGB')
                image_tensor = process_images([image], image_processor, args)
                if type(image_tensor) is list:
                    image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
                else:
                    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
                
                if args.explanation:
                    prompts = [get_scoring_prompt(caption), get_explanation_prompt(caption)]
                else:
                    prompts = [get_scoring_prompt(caption)]
                
                score_smoothing = True

                for prompt in prompts:
                    if image is not None:
                        if model.config.mm_use_im_start_end:
                            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
                        else:
                            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
                        conv.append_message(conv.roles[0], prompt)
                        image = None
                    else:
                        conv.append_message(conv.roles[0], prompt)

                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    
                    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                    
                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                    if args.print_logs and score_smoothing is False:
                        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                    else:
                        streamer = None

                    with torch.inference_mode():
                        if args.print_logs:
                            if score_smoothing:
                                print(f'\nSCORING ({i+1}):')
                            else:
                                print(f'\nEXPLANATION ({i+1}):')

                        output_dict = model.generate(
                                    input_ids,
                                    images=image_tensor,
                                    do_sample=False, # Greedy decoding
                                    num_beams=1, # Greedy decoding
                                    max_new_tokens=512,
                                    streamer=streamer,
                                    use_cache=True,
                                    stopping_criteria=[stopping_criteria],
                                    output_scores=True,
                                    return_dict_in_generate=True,
                                )
                    output_ids = output_dict.sequences
                    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

                    # 1st turn (Scoring stage)
                    if score_smoothing:
                        dotsnumbersdots = re.sub(f'[^\d\.]', '', outputs)
                        numbersdots = re.sub(f'^\.+', '', dotsnumbersdots)
                        numbers = re.sub(r'\.+$', '', numbersdots)
                        score_check = float(numbers)

                        if score_check < 0 or score_check > 1:
                            continue
                        
                        elif score_check < 1.0:
                            num_index_in_score = str(score_check).index('.') + 1
                            find_num = int(str(score_check)[num_index_in_score])
                            num_index_in_token = (output_ids[0,1:] == rate2token[find_num]).nonzero().squeeze()
                            
                            if len(num_index_in_token.shape) > 0:   
                                if find_num == 0:
                                    num_index_in_token = num_index_in_token[1] 
                                else:
                                    num_index_in_token = num_index_in_token[0]
                            probs = output_dict.scores[num_index_in_token]
                            probs = torch.nn.functional.softmax(probs, dim=-1)[0]
                            
                            score = 0.
                            for rate, token in rate2token.items():
                                score += probs[token] * rate * 0.1

                            if len(str(score_check)) > 3:
                                num2_index_in_score = str(score_check).index('.') + 2
                                find_num2 = int(str(score_check)[num2_index_in_score])
                                num2_index_in_token = (output_ids[0,1:] == rate2token[find_num2]).nonzero().squeeze()
                                if len(num2_index_in_token.shape) > 0:
                                    num2_index_in_token = num2_index_in_token[1]
                                probs2 = output_dict.scores[num2_index_in_token]
                                probs2 = torch.nn.functional.softmax(probs2, dim=-1)[0]
                            
                                for rate, token in rate2token.items():
                                    score += probs2[token] * rate * 0.01
                        
                        else:
                            num_index_in_token = (output_ids[0,1:] == rate2token[1]).nonzero().squeeze()
                            probs = output_dict.scores[num_index_in_token]
                            probs = torch.nn.functional.softmax(probs, dim=-1)[0]
                            
                            score = 0.9 * probs[rate2token[0]] + probs[rate2token[1]]
                        
                        conv.messages[-1][-1] = str(score.cpu().item()) + '</s>'
                        if args.print_logs:
                            print(f'Score: ', score.item())

                        item[f'expert_score_{i+1}'] = score.item()

                        score_smoothing = False

                    # 2nd turn (Explanation stage)
                    else:
                        conv.messages[-1][-1] = outputs
                        item[f'explanation_{i+1}'] = parse_explanation(outputs)

            if item['expert_score_1'] > item['expert_score_2']:
                item['expert_pred_label'] = 0
            elif item['expert_score_1'] < item['expert_score_2']:
                item['expert_pred_label'] = 1
            else:
                item['expert_pred_label'] = -1

            item['human_label'] = human_score
            results.append(item)
            
            if args.print_logs:
                print('-'*100)

        if args.explanation:
            result_file = f"{model_name}_exp_{args.input_json[:-5]}_{args.pascal_category.lower()}.json"
        else:
            result_file = f"{model_name}_{args.input_json[:-5]}_{args.pascal_category.lower()}.json"
            
        # Save results
        with open(os.path.join(args.result_fold, result_file), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    compute_metrics(results, args.input_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="hjkim811/EXPERT-llava-13b-lora")
    parser.add_argument("--model-base", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--base_fold", default='annotations')
    parser.add_argument("--input_json", default='flickr8k.json')
    parser.add_argument("--pascal_category", default='HC')
    parser.add_argument("--result_fold", default='results')
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    parser.add_argument("--explanation", type=bool, default=False)
    parser.add_argument("--print_logs", action='store_true')
    args = parser.parse_args()
    main(args)
