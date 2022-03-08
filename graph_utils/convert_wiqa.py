import os
import json
import re
import sys
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

__all__ = ['convert_to_entailment']

# String used to indicate a blank
BLANK_STR = "___"

def convert_to_entailment(qa_file: str, output_file: str, ans_pos: bool=False):
    print(f'converting {qa_file} to entailment dataset...')
    nrow = sum(1 for _ in open(qa_file, 'r'))
    # predictor = Predictor.from_path("ner-elmo.2021-02-12.tar.gz")
    predictor = Predictor.from_path("/home/zhengchen/codes/python_codes/wiqa_research/MRRG/graph_utils/openie-model.2020.03.26.tar.gz") ### link: https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz
    
    path, _ = os.path.split(output_file) ### _ is the filename, path is the folder name
    if not os.path.exists(path):
        os.makedirs(path)

    with open(output_file, 'w+') as output_handle, open(qa_file, 'r') as qa_handle:
        # print("Writing to {} from {}".format(output_file, qa_file))
        for line in tqdm(qa_handle, total=nrow):
            json_line = json.loads(line)
            output_dict = convert_qajson_to_entailment(json_line, predictor)
            output_handle.write(json.dumps(output_dict))
            output_handle.write("\n")
    print(f'converted statements saved to {output_file}')
    print()


### use AllenNLP NER tool
# def convert_qajson_to_entailment(qa_json, predictor):
#     question_text = qa_json["question"]["stem"]
#     para_steps = " ".join(qa_json["question"]['para_steps'])
#     answer_labels = qa_json["question"]["answer_label"]

#     queston_and_para = question_text+para_steps
#     # ner_res = predictor.predict(sentence="Did Uriah honestly think he could beat The Legend of Zelda in under three hours?.")
#     ner_res = predictor.predict(sentence=queston_and_para)
#     statement = ''
#     for i in range(len(ner_res['tags'])):
#         if ner_res['tags'][i] != 'O':
#             statement += ner_res['words'][i] + ' '
#     # statement = create_hypothesis(get_fitb_from_question(question_text), choice_text, ans_pos)
#     create_output_dict(qa_json, statement, True)

#     return qa_json

# predictor = Predictor.from_path("./ner-elmo.2021-02-12.tar.gz")
# convert_qajson_to_entailment('...', predictor)

### use AllenNLP OpenIE tool
def convert_qajson_to_entailment(qa_json, predictor):
    question_text = qa_json["question"]["stem"]
    # para_steps = " ".join(qa_json["question"]['para_steps'])
    para_steps = qa_json["question"]['para_steps']
    answer_labels = qa_json["question"]["answer_label"]

    queston_and_para = [question_text] + para_steps

    set_res = set()
    for sen in queston_and_para:
        res = predictor.predict(sentence=sen.lower())
        for i in range(len(res['verbs'])):
            description = res['verbs'][i]['description']
            start, end= 0, 0
            for i in range(len(description)):
                if description[i] == '[':
                    start = i+1
                elif description[i] == ']':
                    end = i
                    if description[start:end+1].startswith('ARG'):
                        set_res.add(description[start:end].split(':')[1][1:])    

    statement = ', '.join(set_res)
    create_output_dict(qa_json, statement, True)

    return qa_json

# Get a Fill-In-The-Blank (FITB) statement from the question text. E.g. "George wants to warm his
# hands quickly by rubbing them. Which skin surface will produce the most heat?" ->
# "George wants to warm his hands quickly by rubbing them. ___ skin surface will produce the most
# heat?
def get_fitb_from_question(question_text: str) -> str:
    fitb = replace_wh_word_with_blank(question_text)
    if not re.match(".*_+.*", fitb):
        # print("Can't create hypothesis from: '{}'. Appending {} !".format(question_text, BLANK_STR))
        # Strip space, period and question mark at the end of the question and add a blank
        fitb = re.sub(r"[\.\? ]*$", "", question_text.strip()) + " " + BLANK_STR
    return fitb


# Create a hypothesis statement from the the input fill-in-the-blank statement and answer choice.
def create_hypothesis(fitb: str, choice: str, ans_pos: bool) -> str:

    if ". " + BLANK_STR in fitb or fitb.startswith(BLANK_STR):
        choice = choice[0].upper() + choice[1:]
    else:
        choice = choice.lower()
    # Remove period from the answer choice, if the question doesn't end with the blank
    if not fitb.endswith(BLANK_STR):
        choice = choice.rstrip(".")
    # Some questions already have blanks indicated with 2+ underscores
    if not ans_pos:
        hypothesis = re.sub("__+", choice, fitb)
        return hypothesis
    choice = choice.strip()
    m = re.search("__+", fitb)
    start = m.start()

    length = (len(choice) - 1) if fitb.endswith(BLANK_STR) and choice[-1] in ['.', '?', '!'] else len(choice)
    hypothesis = re.sub("__+", choice, fitb)

    return hypothesis, (start, start + length)


# Identify the wh-word in the question and replace with a blank
def replace_wh_word_with_blank(question_str: str):
    # if "What is the name of the government building that houses the U.S. Congress?" in question_str:
    #     print()
    question_str = question_str.replace("What's", "What is")
    question_str = question_str.replace("whats", "what")
    question_str = question_str.replace("U.S.", "US")
    wh_word_offset_matches = []
    wh_words = ["which", "what", "where", "when", "how", "who", "why"]
    for wh in wh_words:
        # Some Turk-authored SciQ questions end with wh-word
        # E.g. The passing of traits from parents to offspring is done through what?

        if wh == "who" and "people who" in question_str:
            continue

        m = re.search(wh + r"\?[^\.]*[\. ]*$", question_str.lower())
        if m:
            wh_word_offset_matches = [(wh, m.start())]
            break
        else:
            # Otherwise, find the wh-word in the last sentence
            m = re.search(wh + r"[ ,][^\.]*[\. ]*$", question_str.lower())
            if m:
                wh_word_offset_matches.append((wh, m.start()))
            # else:
            #     wh_word_offset_matches.append((wh, question_str.index(wh)))

    # If a wh-word is found
    if len(wh_word_offset_matches):
        # Pick the first wh-word as the word to be replaced with BLANK
        # E.g. Which is most likely needed when describing the change in position of an object?
        wh_word_offset_matches.sort(key=lambda x: x[1])
        wh_word_found = wh_word_offset_matches[0][0]
        wh_word_start_offset = wh_word_offset_matches[0][1]
        # Replace the last question mark with period.
        question_str = re.sub(r"\?$", ".", question_str.strip())
        # Introduce the blank in place of the wh-word
        fitb_question = (question_str[:wh_word_start_offset] + BLANK_STR +
                         question_str[wh_word_start_offset + len(wh_word_found):])
        # Drop "of the following" as it doesn't make sense in the absence of a multiple-choice
        # question. E.g. "Which of the following force ..." -> "___ force ..."
        final = fitb_question.replace(BLANK_STR + " of the following", BLANK_STR)
        final = final.replace(BLANK_STR + " of these", BLANK_STR)
        return final

    elif " them called?" in question_str:
        return question_str.replace(" them called?", " " + BLANK_STR + ".")
    elif " meaning he was not?" in question_str:
        return question_str.replace(" meaning he was not?", " he was not " + BLANK_STR + ".")
    elif " one of these?" in question_str:
        return question_str.replace(" one of these?", " " + BLANK_STR + ".")
    elif re.match(r".*[^\.\?] *$", question_str):
        # If no wh-word is found and the question ends without a period/question, introduce a
        # blank at the end. e.g. The gravitational force exerted by an object depends on its
        return question_str + " " + BLANK_STR
    else:
        # If all else fails, assume "this ?" indicates the blank. Used in Turk-authored questions
        # e.g. Virtually every task performed by living organisms requires this?
        return re.sub(r" this[ \?]", " ___ ", question_str)


def create_output_dict(input_json: dict, statement: str, label: bool) -> dict:
    if "statements" not in input_json:
        input_json["statements"] = []
    input_json["statements"].append({"label": True, "statement": statement})
    return input_json


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError("Provide at least two arguments: "
                         "json file with hits, output file name")
    convert_to_entailment(sys.argv[1], sys.argv[2])
