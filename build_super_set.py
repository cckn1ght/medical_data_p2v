import csv
from collections import defaultdict
from gensim.models import Word2Vec
import json


def load_chv(se_dict):
    with open('./CHV_flatfiles_all/CHV_concepts_terms_flatfile_20110204.tsv', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            cui = line[0]
            if cui in se_dict:
                se_dict[cui].add('_'.join(line[1].lower().split()))
                se_dict[cui].add('_'.join(line[2].lower().split()))
                se_dict[cui].add('_'.join(line[2].lower().split()))
        return se_dict


def load_se_and_cui():
    se_dict = defaultdict(set)
    with open('./sider/meddra_all_se.tsv', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            se = '_'.join(line[5].lower().split())
            cui = line[2]
            se_dict[cui].add(se)
    return se_dict


def load_indication_and_cui(se_dict):
    with open('./sider/meddra_all_indications.tsv', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            indi1 = '_'.join(line[3].lower().split())
            indi2 = '_'.join(line[6].lower().split())
            cui = line[1]
            se_dict[cui].add(indi1)
            se_dict[cui].add(indi2)
    return se_dict


def build_set(buildIndication=False):
    super_se_dict = load_se_and_cui()
    if buildIndication:
        super_se_dict = load_indication_and_cui(super_se_dict)
    super_se_dict = load_chv(super_se_dict)
    return super_se_dict


def wirte2file(super_se_dict, buildIndication=False):
    """wirte super set to csv file"""
    if buildIndication:
        fname = './superSetModel/superSESetWithIndication.csv'
    else:
        fname = './superSetModel/superSESet.csv'
    with open(fname, 'w') as f:
        writer = csv.writer(f)
        for cui, seSet in super_se_dict.items():
            for se in seSet:
                writer.writerow([cui, se])


def build_sub_w2v_model(super_se_dict, model):
    total_words = 0
    se_in_vocab = set()
    for se_set in super_se_dict.values():
        for se in se_set:
            try:
                vec = model[se]
                total_words += 1
                se_in_vocab.add(se)
            except KeyError:
                pass
    with open('./superSetModel/super_set_w2v_model', 'w') as f:
        f.write(str(total_words) + ' ' + '200' + '\n')
        for se_set in super_se_dict.values():
            for se in se_set:
                try:
                    vec = ' '.join(map(str, model[se]))
                    f.write(se + ' ' + vec + '\n')
                except KeyError:
                    pass
    return se_in_vocab


def top_n_of_se(se_in_vocab, model, topn=100):
    se_sim_dict = {}
    for se in se_in_vocab:
        se_sim_dict[se] = [(w, sim) for w, sim in model.most_similar(
            se, topn=topn) if sim > 0.5]
    return se_sim_dict


def sim_transpose(se_sim_dict):
    se_sim_transpose = defaultdict(list)
    for key, values in se_sim_dict.items():
        for word, sim in values:
            se_sim_transpose[word].append((key, round(sim, 3)))
    for sim_list in se_sim_transpose.values():
        sim_list.sort(key=lambda x: x[1], reverse=True)
    return se_sim_transpose


def main():
    # model = Word2Vec.load('./cleaned_data/p2vModel')
    model = Word2Vec.load('./cleaned_data/p2vModel_skipG_withSharedTask')
    buildIndication = True
    super_se_dict = build_set(buildIndication)
    sum([len(se_set) for se_set in super_se_dict.values()])
    wirte2file(super_se_dict, buildIndication)
    se_in_vocab = build_sub_w2v_model(super_se_dict, model)
    se_sim_dict = top_n_of_se(se_in_vocab, model)
    se_sim_transpose = sim_transpose(se_sim_dict)
    with open('./cleaned_data/sim_transpose_skipG.json', 'w') as f:
        json.dump(se_sim_transpose, f, indent=2)


if __name__ == '__main__':
    main()
